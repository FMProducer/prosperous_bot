# paper_trader.py

import datetime as dt
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import websocket
from tqdm.auto import tqdm
from sqlalchemy import create_engine, text

# Global cache for SQLAlchemy engines
_engine_cache = {}

def get_engine(dsn: str):
    """
    Creates and caches a SQLAlchemy engine to avoid connection spam during parallel runs.
    """
    if dsn not in _engine_cache:
        # Optuna with multiple jobs creates separate processes.
        # This ensures each process has its own engine, but it's created only once.
        _engine_cache[dsn] = create_engine(dsn)
    return _engine_cache[dsn]

from agent import D3QN_PER_Agent
from config import MasterConfig
from config import cfg as default_cfg
from test_agent import init_agent
from trading_environment import TradingEnvironment, logger
from utils import find_spike_windows, load_config, set_random_seed, setup_logging

class PaperTrader:
    def __init__(self, cfg: MasterConfig, cfg_mod: Any, model_path_override: str = None):
        self.cfg = cfg        
        setup_logging(session_name="paper_trader_session", cfg=self.cfg)
        set_random_seed(self.cfg.random_seed)

        self.agent = self._load_agent(model_path_override)
        self.stats = self._load_stats()

        # --- NEW: Ticker selection logic ---
        if self.cfg.paper.symbols:
            symbols = self.cfg.paper.symbols if isinstance(self.cfg.paper.symbols, list) else [self.cfg.paper.symbols]
            self.symbols_to_trade = symbols if symbols != ["ALL"] else self._get_all_symbols_from_db()
            logging.info(f"Using {len(self.symbols_to_trade)} symbols for paper trading.")
        else:
            try:
                with open("data/tickers.txt", "r") as f:
                    self.symbols_to_trade = [line.strip() for line in f if line.strip()]
                logging.info(f"Config `paper.symbols` is empty. Using {len(self.symbols_to_trade)} symbols from data/tickers.txt")
            except FileNotFoundError:
                self.symbols_to_trade = []
        
        if not self.symbols_to_trade:
            logging.error("No symbols to trade. Exiting.")
            raise SystemExit("Symbol list is empty.")

        self.ws_url = "wss://fstream.binance.com/stream?streams=" + "/".join(
            [f"{s.lower()}@kline_1m" for s in self.symbols_to_trade]
        )
        self.ws: websocket.WebSocketApp = None
        self.ws_thread: threading.Thread = None

        # Data buffers: {symbol: deque(maxlen=...)}
        self.buffer_len = self.cfg.seq.pre_signal_len + self.cfg.seq.post_signal_len + 60  # Add margin
        self.buffers: Dict[str, deque] = {
            symbol: deque(maxlen=self.buffer_len) for symbol in self.symbols_to_trade
        }

        # Trading state
        self.balance = self.cfg.market.initial_balance
        self.open_positions: Dict[str, Dict] = {}
        self.cooldowns: Dict[str, dt.datetime] = {}

        self.simulated_time: dt.datetime = None
        # Metrics
        self.trades_log = []
        self.equity_curve = []

        self._stop_event = threading.Event()

    def _load_agent(self, model_path_override: str) -> D3QN_PER_Agent:
        if model_path_override:
            model_path = model_path_override
            logging.info(f"Using model from command line: {model_path}")
        elif self.cfg.paths.model_path and os.path.exists(self.cfg.paths.model_path):
            model_path = self.cfg.paths.model_path
            logging.info(f"Using model from config file: {model_path}")
        else:
            logging.error("Model path not specified. Please set `cfg.paths.model_path` in your config file.")
            raise FileNotFoundError("Model path not specified in the configuration.")
        return init_agent(model_path, self.cfg, None)  # No cache for live trading

    def _load_stats(self) -> Dict:
        stats_path = self.cfg.paths.norm_stats_path or os.path.join(self.cfg.paths.output_dir, "norm_stats.json")
        if not os.path.exists(stats_path):
            logging.error(f"Normalization stats not found at path: {stats_path}.")
            raise RuntimeError(f"Normalization stats not found at path: {stats_path}.")

        logging.info(f"Loading normalization stats from {stats_path}")
        with open(stats_path, "r") as f:
            return json.load(f)

    def _get_all_symbols_from_db(self) -> List[str]:
        """Fetches all unique symbols from the database."""
        if not self.cfg.db.dsn:
            logging.error("Database DSN `cfg.db.dsn` is not configured.")
            return []
        try:
            engine = get_engine(self.cfg.db.dsn)
            with engine.connect() as conn:
                query = text("SELECT DISTINCT symbol FROM v_klines_1m_npz")
                result = conn.execute(query)
                return [row[0] for row in result]
        except Exception as e:
            logging.error(f"Failed to fetch all symbols from DB for paper trading: {e}", exc_info=True)
            return []

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "stream" not in data:
                return

            kline_data = data["data"]["k"]
            if not kline_data["x"]:  # kline is not closed
                return

            symbol = kline_data["s"]
            kline_ts = dt.datetime.fromtimestamp(kline_data["t"] / 1000, tz=dt.timezone.utc)

            # This is a simplified bar builder. A production one would be more robust.
            new_bar = {
                "ts": kline_ts,
                "open": float(kline_data["o"]),
                "high": float(kline_data["h"]),
                "low": float(kline_data["l"]),
                "close": float(kline_data["c"]),
                "volume": float(kline_data["v"]),
                "volume_weighted_average": float(kline_data["q"]) / (float(kline_data["v"]) + 1e-9),
                "num_trades": int(kline_data["n"]),
            }

            self.buffers[symbol].append(new_bar)
            logging.debug(f"New bar for {symbol} at {kline_ts}. Buffer size: {len(self.buffers[symbol])}")

            # In websocket mode, we check for signals on each new bar
            df = pd.DataFrame(list(self.buffers[symbol])).set_index("ts")
            self._find_and_process_spikes(symbol, df, dt.datetime.now(dt.timezone.utc))

        except Exception as e:
            logging.error(f"Error in _on_message: {e}", exc_info=True)

    def _get_current_time(self) -> dt.datetime:
        """Возвращает симулированное время в режиме БД или реальное время в режиме WebSocket."""
        return self.simulated_time if self.cfg.paper.source == "database" else dt.datetime.now(dt.timezone.utc)

    def _on_open(self, ws):
        logging.info("WebSocket connection opened.")

    def _on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"WebSocket connection closed: {close_status_code} {close_msg}")

    def _on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def _find_and_process_spikes(self, symbol: str, df: pd.DataFrame, current_time: dt.datetime) -> None:
        """Finds spike signals in a dataframe and processes the latest one."""
        now = current_time

        if symbol in self.cooldowns and now < self.cooldowns[symbol]:
            return

        # Find spikes. CRITICAL: use_lookahead=False for live trading
        spike_windows = find_spike_windows(
            df,
            context_minutes=self.cfg.detector.context_minutes,
            window_minutes=self.cfg.detector.window_minutes,
            abs_change_threshold_pct=self.cfg.detector.abs_change_pct,
            contrast_min=self.cfg.detector.contrast_min,
            cooldown_minutes=0,  # Cooldown is managed externally
            use_lookahead=False,
        )

        if not spike_windows:
            return

        # Take the most recent spike signal
        *_, session_start, _, _ = spike_windows[-1]
        signal_dt = session_start

        self._process_signal(symbol, signal_dt, df, current_time)

    def _process_signal(self, symbol: str, signal_dt: dt.datetime, df: pd.DataFrame, current_time: dt.datetime = None) -> None:
        """Processes a single detected signal."""
        now = current_time or signal_dt

        # Check if this signal is new (not within a cooldown period of live trading)
        if symbol in self.cooldowns and now < self.cooldowns[symbol]:
            return # Signal is within cooldown, ignore
        
        logging.info(f"Spike signal detected for {symbol} at {signal_dt}")
        self.cooldowns[symbol] = now + dt.timedelta(minutes=self.cfg.detector.cooldown_minutes)

        # Prepare data for inference
        seq_start = signal_dt - dt.timedelta(minutes=self.cfg.seq.pre_signal_len)
        seq_end = signal_dt + dt.timedelta(minutes=self.cfg.seq.post_signal_len)
        seq_df = df[(df.index >= seq_start) & (df.index < seq_end)]

        if len(seq_df) != self.cfg.seq.full_seq_len:
            logging.warning(f"Could not form full sequence for {symbol} at {signal_dt}. Got {len(seq_df)} rows.")
            return

        # --- FIX START ---
        # Используем список каналов, который мы прописали в alpha_seed_404_pt.py
        # Это гарантирует, что мы берем все 10 колонок (включая taker_base, taker_quote, dummy)
        target_channels = getattr(self.cfg.data, "data_channels", None)
        if target_channels is None:
             target_channels = self.cfg.data.expected_channels # Fallback

        # Проверка на наличие колонок
        missing_cols = [c for c in target_channels if c not in seq_df.columns]
        if missing_cols:
            logging.warning(f"Missing columns in seq_df: {missing_cols}. Available: {seq_df.columns.tolist()}")
            # Если dummy нет, добавим его на лету (хотя мы добавляли в process_historical_chunk)
            if "dummy" in missing_cols:
                 seq_df["dummy"] = 0.0

        session_data = seq_df[target_channels].to_numpy(dtype=np.float32)
        # --- FIX END ---
        # Get action from agent
        action = self._get_agent_action(session_data)

        # entry price with optional execution delay (0 keeps current behavior)
        delay = int(getattr(self.cfg.backtest, "exec_delay_bars", 0))
        delayed_dt = signal_dt + dt.timedelta(minutes=delay)
        if delayed_dt in df.index:
            entry_price = float(df.loc[delayed_dt]["close"])
            entry_dt_used = delayed_dt
        else:
            entry_price = float(df.loc[signal_dt]["close"])
            entry_dt_used = signal_dt

        # Execute trade
        if action in [1, 2]:  # LONG or SHORT
            self._execute_trade(symbol, action, signal_dt, entry_price, entry_dt_used, delay)

    def _process_signal_continuous(self, symbol: str, signal_dt: dt.datetime, df: pd.DataFrame, current_time: dt.datetime = None) -> None:
        """
        Processes continuous trading signal (no spike filter).
        Called on EVERY candle, not just spikes.
        """
        now = current_time or signal_dt
        
        # Cooldown check (to avoid re-entering same symbol too quickly)
        if symbol in self.cooldowns and now < self.cooldowns[symbol]:
            return
        
        # Check if we already have a position for this symbol
        if symbol in self.open_positions:
            return
        
        # Check if we've hit max parallel sessions
        if len(self.open_positions) >= self.cfg.backtest.max_parallel_sessions:
            return
        
        # Prepare sequence for agent
        seq_start = signal_dt - dt.timedelta(minutes=self.cfg.seq.pre_signal_len)
        seq_end = signal_dt
        seq_df = df[(df.index >= seq_start) & (df.index <= seq_end)]
        
        if len(seq_df) != self.cfg.seq.full_seq_len:
            return  # Not enough data yet
        
        # Use configured channels
        target_channels = getattr(self.cfg.data, "data_channels", None)
        if target_channels is None:
            target_channels = self.cfg.data.expected_channels
        
        missing_cols = [c for c in target_channels if c not in seq_df.columns]
        if missing_cols:
            logging.warning(f"Missing columns for {symbol}: {missing_cols}")
            return
        
        session_data = seq_df[target_channels].to_numpy(dtype=np.float32)
        
        # Get action from agent
        action = self._get_agent_action(session_data)
        
        # Only open trade if action is LONG(1) or SHORT(2)
        if action not in [1, 2]:
            return
        
        # Entry price
        entry_price = float(seq_df.iloc[-1]["close"])
        entry_dt_used = signal_dt
        
        # Execute trade
        self._execute_trade(symbol, action, signal_dt, entry_price, entry_dt_used, delay=0)
        
        # Set cooldown
        self.cooldowns[symbol] = now + dt.timedelta(minutes=self.cfg.detector.cooldown_minutes)
    def _get_agent_action(self, session_data: np.ndarray) -> int:
        """Get a trading action from the RL agent."""
        env = TradingEnvironment(
            sequences=[session_data],
            stats=self.stats,
            render_mode=None,
            full_seq_len=self.cfg.seq.full_seq_len,
            num_features=session_data.shape[1],
            num_actions=self.cfg.market.num_actions,
            flat_state_size=self.cfg.seq.flat_state_size,
            initial_balance=self.cfg.market.initial_balance,
            pre_signal_len=self.cfg.seq.pre_signal_len,
            data_channels=self.cfg.data.data_channels,
            slippage=0,
            transaction_fee=0,
            agent_session_len=self.cfg.seq.agent_session_len,
            agent_history_len=self.cfg.seq.agent_history_len,
            input_history_len=self.cfg.seq.input_history_len,
            price_channels=self.cfg.data.price_channels,
            volume_channels=self.cfg.data.volume_channels,
            other_channels=self.cfg.data.other_channels,
            action_history_len=self.cfg.seq.action_history_len,
            inaction_penalty_ratio=0,
            backtest_mode=True,
        )
        obs, _ = env.reset(options={"forced_index": 0})

        # --- NEW: Unified action selection logic from backtest_engine ---
        action = 0 # Default to NOOP
        
        if self.cfg.backtest.selection_strategy == "advantage_based_filter":
            q_vals = self.agent.select_action(state=obs, training=False, return_qvals=True)
            adv = q_vals - q_vals[0]
            potential_action = int(np.argmax(adv))
            
            if potential_action == 1 and adv[potential_action] >= self.cfg.backtest.long_action_threshold:
                action = 1
            elif potential_action == 2 and adv[potential_action] >= self.cfg.backtest.short_action_threshold:
                action = 2

        elif self.cfg.backtest.selection_strategy == "ensemble_q_filter":
            q_mean, q_std = self.agent.predict_ensemble(
                state=obs,
                training=False,
                use_cache=False, # No caching in live/paper trading
                cache_key=None,
                n_samples=self.cfg.backtest.ensemble_n_samples,
            )
            # Приведение к массиву и страховка от скаляра
            q_mean = np.asarray(q_mean, dtype=np.float32)
            if q_mean.ndim == 0:
                logging.warning("predict_ensemble returned scalar q_mean; fallback to select_action(return_qvals=True).")
                q_vals = self.agent.select_action(state=obs, training=False, return_qvals=True)
                q_mean = np.asarray(q_vals, dtype=np.float32)
                q_std = np.zeros_like(q_mean, dtype=np.float32)
            else:
                q_std = np.asarray(q_std, dtype=np.float32) if np.ndim(q_std) else np.zeros_like(q_mean, dtype=np.float32)

            advantage = q_mean - q_mean[0]
            potential_action = int(np.argmax(advantage))
            confidence = advantage[potential_action]
            uncertainty = q_std[potential_action]

            # Check confidence threshold
            confidence_ok = (
                (potential_action == 1 and confidence >= self.cfg.backtest.long_action_threshold) or
                (potential_action == 2 and confidence >= self.cfg.backtest.short_action_threshold)
            )

            # Check uncertainty threshold
            uncertainty_ok = uncertainty < self.cfg.backtest.ensemble_max_sigma

            if confidence_ok and uncertainty_ok:
                action = potential_action
            else:
                logging.debug(f"Action {potential_action} rejected. Confidence OK: {confidence_ok} (conf={confidence:.4f}), Uncertainty OK: {uncertainty_ok} (std={uncertainty:.4f})")

        return action

    def _execute_trade(self, symbol: str, action: int, signal_dt: dt.datetime, entry_price: float, entry_dt_used: dt.datetime, delay: int):
        """Open a paper trade."""
        if symbol in self.open_positions:
            logging.warning(f"Already have an open position for {symbol}. Skipping new trade.")
            return

        if len(self.open_positions) >= self.cfg.backtest.max_parallel_sessions:
            logging.warning("Max parallel sessions reached. Skipping new trade.")
            return

        direction = "LONG" if action == 1 else "SHORT"
        order_size_usdt = getattr(self.cfg.backtest, 'order_size_usdt', 0)

        if order_size_usdt > 0:
            # Safety: don't open a position larger than available balance
            position_size = min(order_size_usdt, self.balance)
            if position_size < order_size_usdt:
                logging.warning(
                    f"order_size_usdt {order_size_usdt:.2f} exceeds balance {self.balance:.2f}; "
                    f"clipped to {position_size:.2f}"
                )
        else:
            position_size = self.balance * self.cfg.backtest.position_fraction

        # --- NEW: Initialize risk management state ---
        rm_state = {}
        if self.cfg.backtest.use_risk_management:
            if direction == "LONG":
                rm_state["trailing_max_price"] = entry_price
            else: # SHORT
                rm_state["trailing_min_price"] = entry_price
            
            # Initialize hysteresis state
            if self.cfg.backtest.delta_p_hysteresis is not None:
                rm_state["p_at_last_tsl_update"] = 0.0

        self.open_positions[symbol] = {
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": signal_dt,
            "size": position_size,
            "close_time": signal_dt + dt.timedelta(minutes=self.cfg.seq.agent_session_len),
            **rm_state
        }
        logging.info(
            f"PAPER TRADE OPEN: {direction} {symbol} at {entry_price:.6f} "
            f"(Size: {position_size:.2f} USDT, delay={delay}, entry_dt={entry_dt_used})"
        )

    def _update_and_close_positions(self):
        """Periodically check and close open positions."""
        now = self._get_current_time()
        symbols_to_close: List[Tuple[str, str, float, dt.datetime]] = [] # (symbol, exit_reason, close_price, close_ts)

        for symbol, pos in list(self.open_positions.items()):
            # Get the correct close price
            if self.cfg.paper.source == "websocket":
                if not self.buffers[symbol]:
                    logging.warning(f"Cannot close position for {symbol}, buffer is empty.")
                    continue
                current_price = self.buffers[symbol][-1]["close"]
                current_ts = self.buffers[symbol][-1]["ts"]
            else: # database mode
                with get_engine(self.cfg.db.dsn).connect() as conn:
                    query = text("SELECT close FROM v_klines_1m_npz WHERE symbol = :symbol AND ts = :ts")
                    result = conn.execute(query, {"symbol": symbol, "ts": int(now.timestamp() * 1000)}).scalar_one_or_none()
                    if result is None:
                        result = conn.execute(query, {"symbol": symbol, "ts": int(pos['close_time'].timestamp() * 1000)}).scalar_one_or_none()
                        if result is None:
                            logging.warning(f"Could not find close price for {symbol} at {now} or {pos['close_time']}. Skipping check.")
                            continue
                    current_price = float(result)
                    current_ts = now

            exit_reason = None
            tsl_price = None
            liquidation_price = None

            # --- Risk Management Logic (as per DIFF.md) ---
            if self.cfg.backtest.use_risk_management:
                d0 = self.cfg.backtest.trailing_stop
                d_min = self.cfg.backtest.trailing_stop_min
                fee = self.cfg.market.transaction_fee
                fee_buf = fee * (self.cfg.backtest.fee_buffer_mult or 2.0)

                if pos["direction"] == "LONG":
                    pos["trailing_max_price"] = max(pos.get("trailing_max_price", current_price), current_price)
                    # Always set a base TSL for symmetric activation, but keep it monotonic (never loosen)
                    base_tsl = pos["trailing_max_price"] * (1 - d0)
                    prev_tsl = pos.get("tsl_price")
                    tsl_price = max(base_tsl, prev_tsl) if prev_tsl is not None else base_tsl

                    if d_min is not None:
                        p = max(0, pos["trailing_max_price"] / pos["entry_price"] - 1)
                        if self.cfg.backtest.delta_p_hysteresis is None or p > pos.get('p_at_last_tsl_update', 0) + self.cfg.backtest.delta_p_hysteresis:
                            if self.cfg.backtest.delta_p_hysteresis is not None:
                                pos['p_at_last_tsl_update'] = p
                            
                            # LONG: без "кепки" относительно entry; до fee_buf держим d0 (эквивалент статическому SL)
                            if p <= fee_buf:
                                d_eff = d0
                            else:
                                d_eff = max(d_min, min(d0, d0 - (p - fee_buf)))
                            advanced_tsl_price = pos["trailing_max_price"] * (1 - d_eff)
                            tsl_price = max(tsl_price, advanced_tsl_price)

                elif pos["direction"] == "SHORT":
                    pos["trailing_min_price"] = min(pos.get("trailing_min_price", current_price), current_price)
                    # Always set a base TSL for symmetric activation, but keep it monotonic (never loosen)
                    base_tsl = pos["trailing_min_price"] * (1 + d0)
                    prev_tsl = pos.get("tsl_price")
                    tsl_price = min(base_tsl, prev_tsl) if prev_tsl is not None else base_tsl

                    if d_min is not None:
                        p = max(0, 1 - pos["trailing_min_price"] / pos["entry_price"])
                        if self.cfg.backtest.delta_p_hysteresis is None or p > pos.get('p_at_last_tsl_update', 0) + self.cfg.backtest.delta_p_hysteresis:
                            if self.cfg.backtest.delta_p_hysteresis is not None:
                                pos['p_at_last_tsl_update'] = p
                            
                            # SHORT: симметрично, без "кепки"; до fee_buf держим d0
                            if p <= fee_buf:
                                d_eff = d0
                            else:
                                d_eff = max(d_min, min(d0, d0 - (p - fee_buf)))
                            advanced_tsl_price = pos["trailing_min_price"] * (1 + d_eff)
                            tsl_price = min(tsl_price, advanced_tsl_price)
                
                if tsl_price is not None:
                    pos['tsl_price'] = tsl_price
            
            # --- NEW: Liquidation Logic for Leveraged Trading ---
            leverage = self.cfg.paper.leverage
            if leverage > 1.0:
                # Цена ликвидации - это цена, при которой убыток достигает 100% от выделенной маржи.
                # PnL% = (current_price / entry_price - 1) * leverage. Loss = -1.
                # (current_price / entry_price - 1) = -1 / leverage
                # current_price = entry_price * (1 - 1 / leverage)
                liquidation_price = pos["entry_price"] * (1 - (1 / leverage)) if pos["direction"] == "LONG" else pos["entry_price"] * (1 + (1 / leverage))


            # --- Unified Position Closing Logic ---
            if pos.get('tsl_price') is not None:
                fee = self.cfg.market.transaction_fee
                if pos["direction"] == "LONG" and current_price <= pos['tsl_price']:
                    break_even_price = pos["entry_price"] * (1 + fee) / (1 - fee)
                    exit_reason = "TSL" if current_price > break_even_price else "TSL SL"
                elif pos["direction"] == "SHORT" and current_price >= pos['tsl_price']:
                    break_even_price = pos["entry_price"] * (1 - fee) / (1 + fee)
                    exit_reason = "TSL" if current_price <= break_even_price else "TSL SL"
            
            # --- NEW: Check for liquidation ---
            if liquidation_price is not None and not exit_reason:
                if pos["direction"] == "LONG" and current_price <= liquidation_price:
                    exit_reason = "LIQUIDATION"
                elif pos["direction"] == "SHORT" and current_price >= liquidation_price:
                    exit_reason = "LIQUIDATION"

            # Time-based exit if no other exit reason was triggered
            if now >= pos["close_time"] and not exit_reason:
                fee = self.cfg.market.transaction_fee
                if pos["direction"] == "LONG":
                    break_even_price = pos["entry_price"] * (1 + fee) / (1 - fee)
                    exit_reason = "TSL Time" if current_price > break_even_price else "Time SL"
                else:  # SHORT
                    break_even_price = pos["entry_price"] * (1 - fee) / (1 + fee)
                    exit_reason = "TSL Time" if current_price <= break_even_price else "Time SL"

            if exit_reason:
                # Symmetric informative event log for TSL/Time/LIQUIDATION
                logging.info(
                    f"PAPER TRADE EVENT: {exit_reason} {pos['direction']} {symbol} at {current_price:.6f} "
                    f"(Entry: {pos['entry_price']:.6f}, Size: {pos['size']:.2f} USDT, ts={current_ts})"
                )
                symbols_to_close.append((symbol, exit_reason, current_price, current_ts))

        # --- Process Closed Symbols ---
        for symbol, exit_reason, current_price, close_ts in symbols_to_close:
            if symbol not in self.open_positions:
                continue
            pos = self.open_positions.pop(symbol)

            if current_price is None:
                logging.warning(f"Could not find close price for {symbol} at {close_ts}. Skipping PnL calculation.")
                continue

            leverage = self.cfg.paper.leverage

            if pos["direction"] == "LONG":
                # Применяем плечо к расчету PnL
                pnl = (current_price - pos["entry_price"]) / pos["entry_price"] * pos["size"] * leverage
            else:
                pnl = (pos["entry_price"] - current_price) / pos["entry_price"] * pos["size"] * leverage
            
            # --- NEW: Handle Liquidation PnL ---
            # При ликвидации убыток равен 100% от выделенной маржи (размера позиции)
            if exit_reason == "LIQUIDATION":
                pnl = -pos["size"]

            # Commission is charged on notional (size × leverage), for both entry and exit
            fees = pos["size"] * self.cfg.paper.leverage * self.cfg.market.transaction_fee * 2
            net_pnl = pnl - fees
            self.balance += net_pnl

            trade_record = {
                "symbol": symbol, "direction": pos["direction"], "entry_time": pos["entry_time"].isoformat(),
                "close_time": close_ts.isoformat(), "entry_price": pos["entry_price"], "close_price": current_price,
                "pnl": net_pnl, "balance": self.balance, "exit_reason": exit_reason,
            }
            self.trades_log.append(trade_record)
            self.equity_curve.append({"ts": close_ts.isoformat(), "balance": self.balance})

            logging.info(
                f"PAPER TRADE CLOSE: {pos['direction']} {symbol} at {current_price:.6f} "
                f"(Entry: {pos['entry_price']:.6f}, Size: {pos['size']:.2f} USDT, Fees: {fees:.2f} USDT, "
                f"PnL: {net_pnl:+.2f} USDT, reason={exit_reason}, close_dt={close_ts}) "
                f"New Balance: {self.balance:.2f} USDT"
            )

    def _run_from_websocket(self):
        """Starts the WebSocket connection and runs the trader in live mode."""
        logging.info(f"Starting trader in 'websocket' mode. Connecting to {self.ws_url}...")
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, name="WebSocketThread")
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Main loop for closing positions
        while not self._stop_event.is_set():
            self._update_and_close_positions()
            time.sleep(5)

    def _run_from_database(self):
        """Optimized continuous simulation."""
        logging.info("Starting CONTINUOUS minute-by-minute database simulation.")
        
        if not hasattr(self.cfg.backtest, "time_range") or not self.cfg.backtest.time_range:
            logging.error("`cfg.backtest.time_range` is not defined. Aborting.")
            return
        
        start_utc = self.cfg.backtest.time_range["start_utc"]
        end_utc = self.cfg.backtest.time_range["end_utc"]
        symbols = self.symbols_to_trade
        
        logging.info(f"Fetching data for {len(symbols)} symbols from {start_utc} to {end_utc}...")
        
        engine = get_engine(self.cfg.db.dsn)
        
        query = text("""
            SELECT ts, symbol, open, high, low, close, volume, 
                   volume_weighted_average, num_trades
            FROM v_klines_1m_npz
            WHERE symbol = ANY(:symbols)
              AND ts >= :start_ts
              AND ts < :end_ts
            ORDER BY ts ASC
        """)
        
        start_ts = int(pd.to_datetime(start_utc).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_utc).timestamp() * 1000)
        
        logging.info(f"Querying DB: ts range [{start_ts}, {end_ts}]")
        
        with engine.connect() as conn:
            df_all = pd.read_sql(query, conn, params={
                "symbols": symbols,
                "start_ts": start_ts,
                "end_ts": end_ts
            })
        
        if df_all.empty:
            logging.warning("No data found.")
            return
        
        logging.info(f"Loaded {len(df_all)} rows. Starting simulation...")
        
        # Добавляем missing колонки
        df_all['taker_base'] = 0.0
        df_all['taker_quote'] = 0.0
        df_all['dummy'] = 0.0
        df_all['ts_dt'] = pd.to_datetime(df_all['ts'], unit='ms', utc=True)
        
        # Группируем
        grouped = df_all.groupby('ts')
        
        # Кэш для проверок
        min_buffer_len = self.cfg.seq.full_seq_len
        symbols_ready = set()  # Символы с достаточной историей
        
        for idx, (current_ts_ms, group) in enumerate(tqdm(grouped, desc="Continuous simulation")):
            current_ts = pd.to_datetime(current_ts_ms, unit='ms', utc=True)
            self.simulated_time = current_ts
            
            # 1. Обновляем позиции
            self._update_and_close_positions_db(group)
            
            # 2. Обновляем буферы
            for _, row in group.iterrows():
                symbol = row['symbol']
                new_bar = {
                    "ts": current_ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "volume_weighted_average": float(row["volume_weighted_average"]),
                    "num_trades": int(row["num_trades"]),
                    "taker_base": 0.0,
                    "taker_quote": 0.0,
                    "dummy": 0.0
                }
                self.buffers[symbol].append(new_bar)
                
                # Отслеживаем готовность символа
                if len(self.buffers[symbol]) >= min_buffer_len:
                    symbols_ready.add(symbol)
            
            # 3. ОПТИМИЗАЦИЯ: пропускаем итерации, где невозможны новые входы
            if len(self.open_positions) >= self.cfg.backtest.max_parallel_sessions:
                continue  # Нет слотов для новых позиций
            
            if not symbols_ready:
                continue  # Буферы ещё не заполнены
            
            # 4. Ищем торговые возможности только среди готовых символов
            for symbol in symbols_ready:
                # Early exit проверки
                if symbol in self.open_positions:
                    continue
                
                if symbol in self.cooldowns and current_ts < self.cooldowns[symbol]:
                    continue
                
                if len(self.open_positions) >= self.cfg.backtest.max_parallel_sessions:
                    break
                
                # Готовим данные ОДИН раз
                buffer_list = list(self.buffers[symbol])
                # Берём последние full_seq_len баров (150, не 90!)
                if len(buffer_list) < min_buffer_len:
                    continue

                recent_bars = buffer_list[-min_buffer_len:]  # 150 баров

                # Формируем sequence БЕЗ DataFrame и БЕЗ транспонирования
                target_channels = getattr(self.cfg.data, "data_channels", None)
                if target_channels is None:
                    target_channels = self.cfg.data.expected_channels

                # Извлекаем данные: форма (time_steps, channels) = (150, 10)
                try:
                    session_data = np.array([
                        [bar[ch] for ch in target_channels]
                        for bar in recent_bars
                    ], dtype=np.float32)
                except KeyError as e:
                    logging.warning(f"Missing channel {e} for {symbol}, skipping")
                    continue

                # НЕ ТРАНСПОНИРУЕМ! TradingEnvironment сделает это сам.
                # Форма: (150, 10) ✅

                # Получаем действие
                action = self._get_agent_action(session_data)

                # Логируем для отладки (первые 500 итераций)
                if idx < 500:
                    logging.info(f"Minute {idx}: {symbol} action={action} at {current_ts}")

                # Только LONG(1) или SHORT(2)
                if action not in [1, 2]:
                    continue

                # Entry price — берем последнюю свечу
                entry_price = float(recent_bars[-1]["close"])

                # Исполняем сделку
                self._execute_trade(symbol, action, current_ts, entry_price, current_ts, delay=0)

                # Cooldown
                self.cooldowns[symbol] = current_ts + dt.timedelta(minutes=self.cfg.detector.cooldown_minutes)
        
        logging.info("Continuous simulation finished.")

    def _save_results(self):
        """Saves the results of the simulation."""
        output_dir = os.path.join(self.cfg.paths.output_dir, "paper_trader")
        os.makedirs(output_dir, exist_ok=True)

        trades_df = pd.DataFrame(self.trades_log)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(output_dir, "paper_trades.csv"), index=False)
            logging.info(f"Saved {len(trades_df)} trades to paper_trades.csv")
        else:
            logging.info("No trades to save.")

        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.to_csv(os.path.join(output_dir, "paper_equity.csv"), index=False)
            logging.info(f"Saved equity curve to paper_equity.csv")
        else:
            logging.info("No equity curve to save.")

    def _process_continuous_step(self, current_ts: int, current_slice_indexed: pd.DataFrame):
        """
        Processes a single minute of data for all symbols in continuous mode.
        """
        current_dt = pd.to_datetime(current_ts, unit='s', utc=True)
        self.simulated_time = current_dt

        # 1. Update/close existing positions based on the new candle's data (high/low for TSL)
        self._update_and_close_positions_db(current_slice_indexed.reset_index())

        # 2. Update data buffers for all symbols in the current slice
        for symbol, row in current_slice_indexed.iterrows():
            if symbol not in self.buffers:
                continue
            
            new_bar = {
                "ts": current_dt,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "volume_weighted_average": float(row.get("vwap", 0.0)),
                "num_trades": int(row["num_trades"]),
                "taker_base": float(row.get("takerbase", 0.0)),
                "taker_quote": float(row.get("takerquote", 0.0)),
                "dummy": float(row.get("dummy", 0.0))
            }
            self.buffers[symbol].append(new_bar)

        # 3. Decide on new trades for symbols without open positions
        for symbol, row in current_slice_indexed.iterrows():
            if symbol in self.open_positions:
                continue
            if symbol in self.cooldowns and current_dt < self.cooldowns[symbol]:
                continue
            if len(self.buffers.get(symbol, [])) < self.cfg.seq.full_seq_len:
                continue

            buffer_list = list(self.buffers[symbol])
            relevant_bars = buffer_list[-self.cfg.seq.full_seq_len:]
            seq_df = pd.DataFrame(relevant_bars).set_index("ts")

            if len(seq_df) != self.cfg.seq.full_seq_len:
                continue

            target_channels = getattr(self.cfg.data, "data_channels", self.cfg.data.expected_channels)
            
            missing_cols = [c for c in target_channels if c not in seq_df.columns]
            if missing_cols:
                logging.warning(f"Continuous mode: Missing columns for {symbol}: {missing_cols}. Filling with 0.")
                for col in missing_cols:
                    seq_df[col] = 0.0

            # --- FIX START: Correct shape and history length ---
            
            # 1. Берем только последние agent_history_len (90), а не full_seq_len (150)
            # Агент обучался смотреть на последние 90 шагов
            agent_hist_len = self.cfg.seq.agent_history_len  # 90
            if len(seq_df) > agent_hist_len:
                seq_df = seq_df.iloc[-agent_hist_len:]
                
            # 2. Преобразуем в numpy
            session_data = seq_df[target_channels].to_numpy(dtype=np.float32)
            
            # 3. ТРАНСПОНИРУЕМ: (Time, Channels) -> (Channels, Time)
            # Агент ожидает форму [C, L]
            session_data = session_data.T  
            
            # --- FIX END ---

            action = self._get_agent_action(session_data)

            if action in [1, 2]:
                signal_dt = current_dt
                entry_price = float(row["close"])
                entry_dt_used = signal_dt
                delay = 0

                self._execute_trade(symbol, action, signal_dt, entry_price, entry_dt_used, delay)
                
                self.cooldowns[symbol] = signal_dt + dt.timedelta(minutes=self.cfg.detector.cooldown_minutes)

    def _update_and_close_positions_db(self, current_slice: pd.DataFrame):
        """
        Checks and closes open positions based on the current data slice.
        This DB-specific version uses the high/low of the candle for more precise TSL simulation.
        """
        now = self.simulated_time
        if not self.open_positions:
            return

        slice_indexed = current_slice.set_index('symbol')
        symbols_to_close: List[Tuple[str, str, float, dt.datetime]] = []  # (symbol, reason, close_price, ts)

        for symbol, pos in list(self.open_positions.items()):
            if symbol not in slice_indexed.index:
                continue

            row = slice_indexed.loc[symbol]
            current_price = float(row['close'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            open_price = float(row['open'])
            current_ts = now

            exit_reason = None
            tsl_price = None
            liquidation_price = None
            close_price_for_exit = current_price # Default close price

            # --- Risk Management Logic (TSL update) ---
            if self.cfg.backtest.use_risk_management:
                d0 = self.cfg.backtest.trailing_stop
                d_min = self.cfg.backtest.trailing_stop_min
                fee = self.cfg.market.transaction_fee
                fee_buf = fee * (self.cfg.backtest.fee_buffer_mult or 2.0)

                if pos["direction"] == "LONG":
                    # Trailing max price is updated based on the high of the current candle
                    pos["trailing_max_price"] = max(pos.get("trailing_max_price", high_price), high_price)
                    base_tsl = pos["trailing_max_price"] * (1 - d0)
                    prev_tsl = pos.get("tsl_price")
                    tsl_price = max(base_tsl, prev_tsl) if prev_tsl is not None else base_tsl

                    if d_min is not None:
                        p = max(0, pos["trailing_max_price"] / pos["entry_price"] - 1)
                        if self.cfg.backtest.delta_p_hysteresis is None or p > pos.get('p_at_last_tsl_update', 0) + self.cfg.backtest.delta_p_hysteresis:
                            if self.cfg.backtest.delta_p_hysteresis is not None:
                                pos['p_at_last_tsl_update'] = p
                            
                            if p <= fee_buf:
                                d_eff = d0
                            else:
                                d_eff = max(d_min, min(d0, d0 - (p - fee_buf)))
                            advanced_tsl_price = pos["trailing_max_price"] * (1 - d_eff)
                            tsl_price = max(tsl_price, advanced_tsl_price)

                elif pos["direction"] == "SHORT":
                    # Trailing min price is updated based on the low of the current candle
                    pos["trailing_min_price"] = min(pos.get("trailing_min_price", low_price), low_price)
                    base_tsl = pos["trailing_min_price"] * (1 + d0)
                    prev_tsl = pos.get("tsl_price")
                    tsl_price = min(base_tsl, prev_tsl) if prev_tsl is not None else base_tsl

                    if d_min is not None:
                        p = max(0, 1 - pos["trailing_min_price"] / pos["entry_price"])
                        if self.cfg.backtest.delta_p_hysteresis is None or p > pos.get('p_at_last_tsl_update', 0) + self.cfg.backtest.delta_p_hysteresis:
                            if self.cfg.backtest.delta_p_hysteresis is not None:
                                pos['p_at_last_tsl_update'] = p
                            
                            if p <= fee_buf:
                                d_eff = d0
                            else:
                                d_eff = max(d_min, min(d0, d0 - (p - fee_buf)))
                            advanced_tsl_price = pos["trailing_min_price"] * (1 + d_eff)
                            tsl_price = min(tsl_price, advanced_tsl_price)
                
                if tsl_price is not None:
                    pos['tsl_price'] = tsl_price

            # --- Liquidation Logic ---
            leverage = self.cfg.paper.leverage
            if leverage > 1.0:
                liquidation_price = pos["entry_price"] * (1 - (1 / leverage)) if pos["direction"] == "LONG" else pos["entry_price"] * (1 + (1 / leverage))

            # --- Position Closing Checks ---
            # Check TSL
            if pos.get('tsl_price') is not None:
                fee = self.cfg.market.transaction_fee
                if pos["direction"] == "LONG" and low_price <= pos['tsl_price']:
                    close_price_for_exit = min(open_price, pos['tsl_price'])
                    break_even_price = pos["entry_price"] * (1 + fee) / (1 - fee)
                    exit_reason = "TSL" if close_price_for_exit > break_even_price else "TSL SL"
                elif pos["direction"] == "SHORT" and high_price >= pos['tsl_price']:
                    close_price_for_exit = max(open_price, pos['tsl_price'])
                    break_even_price = pos["entry_price"] * (1 - fee) / (1 + fee)
                    exit_reason = "TSL" if close_price_for_exit <= break_even_price else "TSL SL"
            
            # Check Liquidation
            if liquidation_price is not None and not exit_reason:
                if pos["direction"] == "LONG" and low_price <= liquidation_price:
                    exit_reason = "LIQUIDATION"
                    close_price_for_exit = min(open_price, liquidation_price)
                elif pos["direction"] == "SHORT" and high_price >= liquidation_price:
                    exit_reason = "LIQUIDATION"
                    close_price_for_exit = max(open_price, liquidation_price)

            # Check Time-based exit
            if now >= pos["close_time"] and not exit_reason:
                close_price_for_exit = current_price
                fee = self.cfg.market.transaction_fee
                if pos["direction"] == "LONG":
                    break_even_price = pos["entry_price"] * (1 + fee) / (1 - fee)
                    exit_reason = "TSL Time" if close_price_for_exit > break_even_price else "Time SL"
                else:  # SHORT
                    break_even_price = pos["entry_price"] * (1 - fee) / (1 + fee)
                    exit_reason = "TSL Time" if close_price_for_exit <= break_even_price else "Time SL"

            if exit_reason:
                logging.info(
                    f"PAPER TRADE EVENT: {exit_reason} {pos['direction']} {symbol} at {close_price_for_exit:.6f} "
                    f"(Entry: {pos['entry_price']:.6f}, Size: {pos['size']:.2f} USDT, ts={current_ts})"
                )
                symbols_to_close.append((symbol, exit_reason, close_price_for_exit, current_ts))

        # --- Process Closed Symbols ---
        for symbol, exit_reason, close_price, close_ts in symbols_to_close:
            if symbol not in self.open_positions:
                continue
            pos = self.open_positions.pop(symbol)

            leverage = self.cfg.paper.leverage
            if pos["direction"] == "LONG":
                pnl = (close_price - pos["entry_price"]) / pos["entry_price"] * pos["size"] * leverage
            else:
                pnl = (pos["entry_price"] - close_price) / pos["entry_price"] * pos["size"] * leverage
            
            if exit_reason == "LIQUIDATION":
                pnl = -pos["size"]

            fees = pos["size"] * self.cfg.paper.leverage * self.cfg.market.transaction_fee * 2
            net_pnl = pnl - fees
            self.balance += net_pnl

            trade_record = {
                "symbol": symbol, "direction": pos["direction"], "entry_time": pos["entry_time"].isoformat(),
                "close_time": close_ts.isoformat(), "entry_price": pos["entry_price"], "close_price": close_price,
                "pnl": net_pnl, "balance": self.balance, "exit_reason": exit_reason,
            }
            self.trades_log.append(trade_record)
            self.equity_curve.append({"ts": close_ts.isoformat(), "balance": self.balance})

            logging.info(
                f"PAPER TRADE CLOSE: {pos['direction']} {symbol} at {close_price:.6f} "
                f"(Entry: {pos['entry_price']:.6f}, Size: {pos['size']:.2f} USDT, Fees: {fees:.2f} USDT, "
                f"PnL: {net_pnl:+.2f} USDT, reason={exit_reason}, close_dt={close_ts}) "
                f"New Balance: {self.balance:.2f} USDT"
            )

    def run(self):
        """Start the paper trader based on the configured source."""
        try:
            if self.cfg.paper.source == "websocket":
                self._run_from_websocket()
            elif self.cfg.paper.source == "database":
                self._run_from_database()
            else:
                logging.error(f"Unknown paper trader source: '{self.cfg.paper.source}'")
        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully shut down the paper trader."""
        logging.info("Shutting down Paper Trader...")
        self._stop_event.set()
        if self.ws:
            self.ws.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)

        # Save metrics
        output_dir = os.path.join(self.cfg.paths.output_dir, "paper_trader")
        os.makedirs(output_dir, exist_ok=True)

        trades_df = pd.DataFrame(self.trades_log)
        trades_df.to_csv(os.path.join(output_dir, "paper_trades.csv"), index=False)
        logging.info(f"Saved {len(trades_df)} trades to paper_trades.csv")

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(os.path.join(output_dir, "paper_equity.csv"), index=False)
        logging.info(f"Saved equity curve to paper_equity.csv")

        logging.info("Shutdown complete.")


class PaperTraderDB(PaperTrader):
    """
    Расширенная версия для симуляции торговли на исторических данных из БД.
    Эта версия выполняет симуляцию "свеча за свечой" для более точного моделирования.
    """
    
    def run_db_simulation(self, start_time: str, end_time: str):
        """
        Запускает симуляцию на данных из БД построчно (candle-by-candle).
        """
        logging.info(f"Starting DB candle-by-candle simulation from {start_time} to {end_time}")
        
        engine = get_engine(self.cfg.db.dsn)
        symbols = self.symbols_to_trade
        
        # 1. Загружаем все данные за период одним запросом (или чанками, если памяти мало)
        # Для оптимизации лучше грузить только нужные колонки
        query = text("""
            SELECT open_time_ms as ts, symbol, 
                   open_price as open, 
                   high_price as high, 
                   low_price as low, 
                   close_price as close, 
                   base_volume as volume, 
                   quote_volume as q, 
                   trade_count as n,
                   taker_base, 
                   taker_quote,
                   0.0 as dummy   -- Заглушка для 10-го канала
            FROM klines_1m 
            WHERE open_time_ms >= :start_ts AND open_time_ms <= :end_ts 
            AND symbol = ANY(:symbols)
            ORDER BY open_time_ms ASC
        """)
        
        # Конвертируем даты в timestamp ms
        t_start = int(pd.Timestamp(start_time).timestamp() * 1000)
        t_end = int(pd.Timestamp(end_time).timestamp() * 1000)
        
        logging.info("Fetching historical data for candle-by-candle simulation...")
        with engine.connect() as conn:
            # Используем stream или chunks для больших объемов
            df_iter = pd.read_sql(query, conn, params={
                "start_ts": t_start, 
                "end_ts": t_end, 
                "symbols": symbols
            }, chunksize=100000) # Chunk size can be tuned
            
            for chunk_df in df_iter:
                self._process_historical_chunk(chunk_df)
                
        self.shutdown()

    def _process_historical_chunk(self, df: pd.DataFrame):
        """
        Проходит по чанку данных, группируя по времени, чтобы имитировать "тики" рынка,
        следуя правильному порядку операций.
        """
        # Группируем по времени, чтобы подавать данные "срезами" по всем монетам сразу
        grouped = df.groupby('ts')

        for ts_ms, group in tqdm(grouped, desc="Simulating candles"):
            current_dt = dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc)
            self.simulated_time = current_dt

            # 1. Сначала проверяем выходы (Stop Loss / Take Profit) на основе данных текущей группы свечей
            # Это имитирует то, что произошло ВНУТРИ этой минуты
            self._update_and_close_positions_db(group) # <-- Переместить вызов сюда, ДО цикла обновления буферов

            # 2. Затем обновляем буферы (добавляем свечу как "свершившуюся")
            for _, row in group.iterrows():
                symbol = row['symbol']
                new_bar = {
                    "ts": current_dt,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "volume_weighted_average": float(row["q"]) / (float(row["volume"]) + 1e-9),
                    "num_trades": int(row["n"]),
                    "taker_base": float(row["taker_base"]),
                    "taker_quote": float(row["taker_quote"]),
                    "dummy": 0.0
                }
                self.buffers[symbol].append(new_bar)

                # 3. Ищем сигналы для входа (на основе уже обновленной истории)
                if len(self.buffers[symbol]) >= self.cfg.seq.full_seq_len:
                    df_buffer = pd.DataFrame(list(self.buffers[symbol])).set_index("ts")
                    self._find_and_process_spikes(symbol, df_buffer, current_dt)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/alpha.py"
    model_path_arg = sys.argv[2] if len(sys.argv) > 2 else None

    import importlib.util

    if config_path:
        spec = importlib.util.spec_from_file_location("experiment_cfg", config_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        cfg = cfg_mod.cfg
    else:
        cfg_mod = None
        cfg = default_cfg

    trader = PaperTrader(cfg=cfg, cfg_mod=None, model_path_override=model_path_arg)
    trader.run()