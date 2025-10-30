# backtest_engine.py

import datetime as dt
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
from psycopg2.extras import RealDictCursor

from config import MasterConfig
from config import cfg as default_cfg
from test_agent import init_agent
from trading_environment import TradingEnvironment
from utils import (
    calculate_normalization_stats,
    create_signal_groups,
    find_spike_windows,
    load_config,
    load_npz_dataset_keys,
    load_npz_dataset,
    select_and_arrange_channels,
    set_random_seed,
)


def setup_logging(cfg: MasterConfig) -> None:
    """
    Smart logger setup for backtesting: safe for multiple calls,
    creates 'backtest_session.log' only if it doesn't already exist.
    """
    log_dir = cfg.paths.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "backtest_session.log")

    logger = logging.getLogger()

    # Only configure logging if no handlers are present. This prevents overriding parent script's logging.
    # When run from optimize_cfg with multiprocessing, each process is new.
    # We check if the root logger has handlers. If so, a parent process configured it.
    if logging.getLogger().hasHandlers():
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info("[Init] Logging for backtest session started")


class TradeSummary:
    def __init__(self):
        self.trade_records = []

    def log_trade(self, info: dict, balance: float):
        ticker = info.get("ticker")
        trade_dt = info.get("trade_dt")
        direction = info.get("direction")
        trade_amount = info.get("trade_amount")
        pnl = info.get("trade_realized_pnl")
        change_pct = (pnl / trade_amount) * 100 if trade_amount else 0.0
        balance_pct = (pnl / balance) * 100 if balance else 0.0
        price_delta_pct = info.get("trade_price_delta") * 100

        trade_result = (
            f": {trade_dt.strftime('%Y-%m-%d %H:%M')} {direction:<5} {ticker:<12} {int(trade_amount):>6}:"
            f"   {pnl:+7.2f} ({change_pct:+6.2f}%  |{balance_pct:+7.2f}%) PRICE CHANGE: {price_delta_pct:+.2f}%"
        )
        self.trade_records.append(trade_result)

    def dump(self):
        for trade_result in self.trade_records:
            logging.info(trade_result)


class MetricsCollector:
    def __init__(self):
        self.pnl_by_day: Dict[dt.date, float] = defaultdict(float)
        self.pnl_all = []
        self.changes = []
        self.drawdowns = []
        self.trade_amounts = []
        self.balance_curve: Dict[dt.datetime, Tuple[dt.datetime, float]] = defaultdict()
        self.total_commission = 0.0
        self.correct_preds = 0
        self.total_trades = 0
        self.total_longs = 0
        self.total_shorts = 0
        self.correct_longs = 0
        self.correct_shorts = 0

    def update(self, signal_dt: dt.datetime, info: dict, balance: float):
        pnl = info.get("trade_realized_pnl")
        commission = info.get("total_commission")
        price_change = info.get("trade_price_delta")
        drawdown = info.get("max_drawdown")
        amount = info.get("trade_amount")
        direction = info.get("direction")
        correct = info.get("correct_prediction")

        self.pnl_by_day[signal_dt.date()] += pnl
        self.pnl_all.append(pnl)
        self.changes.append(price_change)
        self.drawdowns.append(drawdown)
        self.trade_amounts.append(amount)
        self.total_commission += commission
        self.total_trades += 1

        if direction == "LONG":
            self.total_longs += 1
            if correct:
                self.correct_longs += 1
        elif direction == "SHORT":
            self.total_shorts += 1
            if correct:
                self.correct_shorts += 1

        if correct:
            self.correct_preds += 1

        self.balance_curve[signal_dt] = (signal_dt, balance)

    def finalize(self):
        pnl_all = np.array(self.pnl_all)
        pnl_by_day = np.array(list(self.pnl_by_day.values()))
        changes = np.array(self.changes)

        if not self.balance_curve:
            return {}

        _, balances = zip(*sorted(self.balance_curve.values()))
        total_change = balances[-1] / balances[0] if balances[0] != 0 else 1.0
        trade_days = len(pnl_by_day)

        std_pnl_by_day_neg = pnl_by_day[pnl_by_day < 0].std() if np.any(pnl_by_day < 0) else 0.0
        std_pnl_all_neg = pnl_all[pnl_all < 0].std() if np.any(pnl_all < 0) else 0.0

        return {
            "total_commission": f"{(-self.total_commission / balances[0]) * 100:.2f}%" if balances[0] != 0 else "0.00%",
            "avg_commission": f"{-self.total_commission / self.total_trades:.2f}" if self.total_trades > 0 else "0.00",
            "max_loss": f"{pnl_all.min():.2f}" if len(pnl_all) > 0 else "0.00",
            "max_profit": f"{pnl_all.max():.2f}" if len(pnl_all) > 0 else "0.00",
            "total_trade_days": trade_days,
            "profit_days": (
                f"{int((pnl_by_day > 0).sum())} ({(pnl_by_day > 0).sum() / trade_days * 100:.2f}%)"
                if trade_days > 0
                else "0 (0.00%)"
            ),
            "final_balance_change": f"{(total_change - 1) * 100:.2f}%",
            "exp_day_change": (
                f"{(np.power(total_change, 1 / trade_days) - 1) * 100:.2f}%" if trade_days > 0 else "0.00%"
            ),
            "max_drawdown": f"{min(self.drawdowns) * 100:.2f}%" if self.drawdowns else "0.00%",
            "sharpe": (
                f"{(pnl_by_day.mean() / (pnl_by_day.std() + 1e-9)) * np.sqrt(len(pnl_by_day)):.2f}"
                if len(pnl_by_day) > 0
                else "0.00"
            ),
            "sortino": (
                f"{(pnl_by_day.mean() / std_pnl_by_day_neg) * np.sqrt(len(pnl_by_day)):.2f}"
                if len(pnl_by_day) > 0 and std_pnl_by_day_neg > 1e-9
                else "0.00"
            ),
            "trades_sharpe": (f"{(pnl_all.mean() / (pnl_all.std() + 1e-9)):.2f}" if len(pnl_all) > 0 else "0.00"),
            "trades_sortino": (
                f"{(pnl_all.mean() / std_pnl_all_neg):.2f}" if len(pnl_all) > 0 and std_pnl_all_neg > 1e-9
                else "0.00"
            ),
            "accuracy": (f"{self.correct_preds / self.total_trades * 100:.1f}%" if self.total_trades > 0 else "0.0%"),
            "total_trades": self.total_trades,
            "total_longs": self.total_longs,
            "total_shorts": self.total_shorts,
            "longs_correct": (
                f"{self.correct_longs} (0.0%)"
                if self.total_longs == 0
                else f"{self.correct_longs} ({(self.correct_longs / self.total_longs) * 100:.1f}%)"
            ),
            "shorts_correct": (
                f"{self.correct_shorts} (0.0%)"
                if self.total_shorts == 0
                else f"{self.correct_shorts} ({(self.correct_shorts / self.total_shorts) * 100:.1f}%)"
            ),
            "correct_avg_change": (f"{np.mean(changes[changes > 0]) * 100:.2f}%" if np.any(changes > 0) else "0.00%"),
            "correct_std_change": (f"{np.std(changes[changes > 0]) * 100:.2f}%" if np.any(changes > 0) else "0.00%"),
            "incorrect_avg_change": (
                f"{np.mean(changes[changes <= 0]) * 100:.2f}%" if np.any(changes <= 0) else "0.00%"
            ),
            "incorrect_std_change": (
                f"{np.std(changes[changes <= 0]) * 100:.2f}%" if np.any(changes <= 0) else "0.00%"
            ),
            "avg_trade_amount": (f"{np.mean(self.trade_amounts):.2f}" if len(self.trade_amounts) > 0 else "0.00"),
            "trades_per_day": (f"{self.total_trades / trade_days:.2f}" if trade_days > 0 else "0.00"),
        }

    def plot_balance(self, path: str):
        if not self.balance_curve:
            return
        times, balances = zip(*sorted(self.balance_curve.values()))
        plt.figure(figsize=(12, 6))
        plt.plot(times, balances, label="Balance", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.title("Balance Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()


def get_pass_advantage(action: int, confidence: float, cfg: MasterConfig) -> bool:
    long_pass = action == 1 and confidence <= cfg.backtest.long_action_threshold
    short_pass = action == 2 and confidence <= cfg.backtest.short_action_threshold
    close_pass = action == 3 and confidence <= cfg.backtest.close_action_threshold
    pass_adv = long_pass or short_pass or close_pass
    return pass_adv


def load_from_db_and_prepare_signals(cfg: MasterConfig) -> List[Tuple[Tuple[str, dt.datetime], np.ndarray]]:
    """
    Dynamically finds spike signals in the database for the given symbols and time range.
    This is a high-performance version that loads data once and processes it in memory.
    """
    if not hasattr(cfg.backtest, "time_range") or not cfg.backtest.time_range:
        logging.error("`cfg.backtest.time_range` is not defined in the config. Aborting.")
        raise ValueError("Backtest time range must be specified in the configuration.")

    start_utc = cfg.backtest.time_range["start_utc"]
    end_utc = cfg.backtest.time_range["end_utc"]
    
    engine = create_engine(cfg.db.dsn)

    if isinstance(cfg.paper.symbols, list) and cfg.paper.symbols:
        symbols = cfg.paper.symbols
    elif cfg.paper.symbols == "ALL":
        logging.info("`paper.symbols` is 'ALL'. Fetching all available symbols from the database for the given time range...")
        try:
            with engine.connect() as conn:
                from sqlalchemy import text
                query = text("SELECT DISTINCT symbol FROM v_klines_1m_npz WHERE ts >= :start_ts AND ts < :end_ts")
                result = conn.execute(query, {
                    "start_ts": int(pd.to_datetime(start_utc).timestamp() * 1000),
                    "end_ts": int(pd.to_datetime(end_utc).timestamp() * 1000)
                })
                symbols = [row[0] for row in result]
        except Exception as e:
            logging.error(f"Failed to fetch all symbols from DB: {e}", exc_info=True)
            return []
    else:
        try:
            with open("data/tickers.txt", "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error("`paper.symbols` is empty and data/tickers.txt not found. No symbols to trade.")
            return []

    # --- Step 1: Find all spike moments using an efficient SQL query ---
    logging.info(f"Scanning for spike signals from {start_utc} to {end_utc} for {len(symbols)} symbols...")
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            detector_cfg = cfg.detector
            logging.info(f"[Detector] use_lookahead={detector_cfg.use_lookahead} "
                         f"window_minutes={detector_cfg.window_minutes} "
                         f"context_minutes={detector_cfg.context_minutes}")

            # ДВА ВАРИАНТА SQL: без параметризации булевой ветки
            if detector_cfg.use_lookahead:
                query = text(f"""
                WITH minute_returns AS (
                    SELECT
                        ts,
                        symbol,
                        close,
                        (close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS ret
                    FROM v_klines_1m_npz
                    WHERE symbol = ANY(:symbols) AND ts >= :start_ts AND ts < :end_ts
                ),
                rolling_stats AS (
                    SELECT
                        ts,
                        symbol,
                        (LEAD(close, {detector_cfg.window_minutes}) OVER (PARTITION BY symbol ORDER BY ts) / close) - 1 AS abs_change,
                        AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY ts
                            ROWS BETWEEN {detector_cfg.context_minutes} PRECEDING AND 1 PRECEDING) AS avg_abs_ret_pre
                    FROM minute_returns
                )
                SELECT ts, symbol
                FROM rolling_stats
                WHERE
                    ABS(abs_change) * 100.0 >= :abs_change_pct AND
                    (ABS(abs_change) / (avg_abs_ret_pre + 1e-9)) >= :contrast_min
                ORDER BY ts, symbol;
                """)
            else:
                query = text(f"""
                WITH minute_returns AS (
                    SELECT
                        ts,
                        symbol,
                        close,
                        (close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS ret
                    FROM v_klines_1m_npz
                    WHERE symbol = ANY(:symbols) AND ts >= :start_ts AND ts < :end_ts
                ),
                rolling_stats AS (
                    SELECT
                        ts,
                        symbol,
                        (close / LAG(close, {detector_cfg.window_minutes}) OVER (PARTITION BY symbol ORDER BY ts)) - 1 AS abs_change,
                        AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY ts
                            ROWS BETWEEN {detector_cfg.context_minutes + detector_cfg.window_minutes} PRECEDING
                            AND {detector_cfg.window_minutes} PRECEDING) AS avg_abs_ret_pre
                    FROM minute_returns
                )
                SELECT ts, symbol
                FROM rolling_stats
                WHERE
                    ABS(abs_change) * 100.0 >= :abs_change_pct AND
                    (ABS(abs_change) / (avg_abs_ret_pre + 1e-9)) >= :contrast_min
                ORDER BY ts, symbol;
                """)

            found_spikes_df = pd.read_sql(query, conn, params={
                "symbols": symbols,
                "start_ts": int(pd.to_datetime(start_utc).timestamp() * 1000),
                "end_ts": int(pd.to_datetime(end_utc).timestamp() * 1000),
                "abs_change_pct": detector_cfg.abs_change_pct,
                "contrast_min": detector_cfg.contrast_min,
            })
            found_spikes_df['ts'] = pd.to_datetime(found_spikes_df['ts'], unit='ms', utc=True)
    except Exception as e:
        logging.error(f"Failed to scan for spikes in database: {e}", exc_info=True)
        return []

    if found_spikes_df.empty:
        logging.warning("No spike signals found in the database for the given period.")
        return []

    logging.info(f"Found {len(found_spikes_df)} potential spike signals. Applying cooldowns...")

    # --- Step 2: Apply cooldown logic in Python ---
    all_signals = []
    last_signal_time = {}
    for index, row in found_spikes_df.iterrows():
        symbol, signal_dt = row['symbol'], row['ts']
        if signal_dt > last_signal_time.get(symbol, dt.datetime.min.replace(tzinfo=dt.timezone.utc)):
            all_signals.append((symbol, signal_dt))
            last_signal_time[symbol] = signal_dt + dt.timedelta(minutes=detector_cfg.cooldown_minutes)

    # --- Step 3: Load data only for the filtered signals ---
    logging.info(f"After cooldown, {len(all_signals)} signals remain. Loading session data...")
    backtest_raw = []
    with engine.connect() as conn:
        for symbol, signal_dt in tqdm(all_signals, desc="Loading signal data"):
            seq_start = signal_dt - dt.timedelta(minutes=cfg.seq.pre_signal_len)
            seq_end = signal_dt + dt.timedelta(minutes=cfg.seq.post_signal_len)
            
            query = text(
                "SELECT ts, open, high, low, close, volume, volume_weighted_average, num_trades "
                "FROM v_klines_1m_npz WHERE symbol = :symbol AND ts >= :start_ts AND ts < :end_ts ORDER BY ts ASC;"
            )
            df_signal = pd.read_sql(query, conn, params={
                "symbol": symbol,
                "start_ts": int(seq_start.timestamp() * 1000),
                "end_ts": int(seq_end.timestamp() * 1000)
            })

            if len(df_signal) == cfg.seq.full_seq_len:
                seq_arr = df_signal[cfg.data.expected_channels].to_numpy(dtype=np.float32)
                backtest_raw.append(((symbol, signal_dt), seq_arr))

    logging.info(f"Dynamically found {len(backtest_raw)} total signals.")
    return backtest_raw

def load_from_db_using_npz_keys(cfg: MasterConfig) -> List[Tuple[Tuple[str, dt.datetime], np.ndarray]]:
    """
    Loads session data from the database, but uses the exact signal keys (symbol, datetime)
    from the reference NPZ file to ensure perfect alignment for backtest comparison.
    """
    npz_keys = load_npz_dataset_keys(cfg.paths.backtest_data_path)
    logging.info(f"Loaded {len(npz_keys)} signal keys from {cfg.paths.backtest_data_path} for DB loading.")

    dsn = cfg.db.dsn
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    backtest_raw = []
    for symbol, signal_dt in tqdm(npz_keys, desc="Loading sessions from DB using NPZ keys"):
        seq_start_dt = signal_dt - dt.timedelta(minutes=cfg.seq.pre_signal_len)
        seq_end_dt = signal_dt + dt.timedelta(minutes=cfg.seq.post_signal_len)
        start_ts = int(seq_start_dt.timestamp() * 1000)
        end_ts = int(seq_end_dt.timestamp() * 1000)

        query = (
            "SELECT ts, open, high, low, close, volume, volume_weighted_average, num_trades "
            "FROM v_klines_1m_npz WHERE symbol = %s AND ts >= %s AND ts < %s ORDER BY ts ASC;"
        )
        cur.execute(query, (symbol, start_ts, end_ts))
        rows = cur.fetchall()

        if len(rows) == cfg.seq.full_seq_len:
            df = pd.DataFrame(rows)
            seq_arr = df[cfg.data.expected_channels].to_numpy(dtype=np.float32)
            backtest_raw.append(((symbol, signal_dt), seq_arr))
        else:
            logging.warning(f"Skipping key {(symbol, signal_dt)}: incorrect data length from DB ({len(rows)}).")

    conn.close()
    logging.info(f"Successfully constructed {len(backtest_raw)} sessions from DB.")
    return backtest_raw


def run_backtest(cfg: MasterConfig, model_path_override: str = None) -> Dict[str, Any]:
    cfg.backtest_mode = True
    setup_logging(cfg)
    set_random_seed(cfg.random_seed)

    data_source = cfg.backtest.data_source
    logging.info(f"Backtest data source: '{data_source}'")

    if data_source == "find_spikes":
        backtest_raw = load_from_db_and_prepare_signals(cfg)
    elif data_source == "npz_keys" and os.path.exists(cfg.paths.backtest_data_path):
        backtest_raw = load_npz_dataset(cfg.paths.backtest_data_path, "Backtest", cfg.paths.plot_dir, cfg.debug.debug_max_size_data)
    else:
        logging.error(f"Data source '{data_source}' not found or file '{cfg.paths.backtest_data_path}' is missing.")
        return {}

    if not backtest_raw:
        logging.error("No data to backtest. Exiting.")
        return {}

    grouped_backtest_data = create_signal_groups(backtest_raw)

    # --- NEW: Логика загрузки/сохранения статистик нормализации ---
    stats_path = cfg.paths.norm_stats_path or os.path.join(cfg.paths.output_dir, "norm_stats.json")
    stats = None
    if os.path.exists(stats_path):
        logging.info(f"Loading normalization stats from {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)

    if stats is None:
        logging.error(f"Normalization stats not found at path: {stats_path}. Please generate them first.")
        raise RuntimeError(f"Normalization stats not found at path: {stats_path}.")

    if model_path_override:
        model_path = model_path_override
        logging.info(f"Using model from command line: {model_path}")
    elif cfg.paths.model_path and os.path.exists(cfg.paths.model_path):
        model_path = cfg.paths.model_path
        logging.info(f"Using model from config file: {model_path}")
    else:
        logging.error("Model path not specified. Please set `cfg.paths.model_path` in your config file.")
        raise FileNotFoundError("Model path not specified in the configuration.")

    agent = init_agent(model_path, cfg, cfg.paths.extra_cache_dir or cfg.paths.cache_dir)

    if cfg.backtest.clear_disk_cache:
        agent.clear_disk_cache()

    result = MetricsCollector()
    trade_log = TradeSummary()
    balance = cfg.market.initial_balance
    open_sessions: List[Dict] = []

    logging.info("\n[Starting backtest...]:")
    thresholds = [
        cfg.backtest.long_action_threshold,
        cfg.backtest.short_action_threshold,
        cfg.backtest.close_action_threshold,
    ]

    for signal_dt, signals in grouped_backtest_data.items():
        open_sessions = [open_s for open_s in open_sessions if open_s["end_time"] > signal_dt]
        free_slots = cfg.backtest.max_parallel_sessions - len(open_sessions)
        if free_slots <= 0:
            logging.info("Too many tickers received, skipping")
            continue

        selected_signals = signals[:free_slots]
        logging.info(
            f": Got {len(signals)} signals @ Date: {signal_dt.date()} Time: {signal_dt.strftime('%H:%M')} For Tickers -> {', '.join(t for t, _ in signals)}"
        )

        for ticker_name, session in selected_signals:
            position_size = balance * cfg.backtest.position_fraction

            env = TradingEnvironment(
                sequences=[session],
                stats=stats,
                render_mode=cfg.render_mode,
                full_seq_len=cfg.seq.full_seq_len,
                num_features=cfg.seq.num_features,
                num_actions=cfg.market.num_actions,
                flat_state_size=cfg.seq.flat_state_size,
                initial_balance=position_size,
                pre_signal_len=cfg.seq.pre_signal_len,
                data_channels=cfg.data.data_channels,
                slippage=cfg.market.slippage,
                transaction_fee=cfg.market.transaction_fee,
                agent_session_len=cfg.seq.agent_session_len,
                agent_history_len=cfg.seq.agent_history_len,
                input_history_len=cfg.seq.input_history_len,
                price_channels=cfg.data.price_channels,
                volume_channels=cfg.data.volume_channels,
                other_channels=cfg.data.other_channels,
                action_history_len=cfg.seq.action_history_len,
                inaction_penalty_ratio=cfg.market.inaction_penalty_ratio,
                backtest_mode=cfg.backtest_mode,
                use_risk_management=cfg.backtest.use_risk_management,
            )

            # align execution timing with config (0 keeps current behavior; 1 = honest next-bar execution)
            env.exec_delay_bars = getattr(cfg.backtest, "exec_delay_bars", 0)
            logging.info(f"[Backtest] exec_delay_bars={env.exec_delay_bars}")

            obs, _ = env.reset()
            for step in range(cfg.seq.agent_session_len):
                cache_key = (ticker_name, signal_dt + dt.timedelta(minutes=step))
                if cfg.backtest.selection_strategy == "advantage_based_filter":
                    q_vals = agent.select_action(
                        state=obs,
                        training=False,
                        return_qvals=cfg.backtest.return_qvals,
                        use_cache=cfg.backtest.use_cache,
                        cache_key=cache_key,
                    )
                    adv = q_vals - q_vals[0]
                    action = int(np.argmax(adv))

                    # Correctly check if the action's confidence meets the threshold
                    if action == 1 and adv[action] < cfg.backtest.long_action_threshold:
                        action = 0
                    elif action == 2 and adv[action] < cfg.backtest.short_action_threshold:
                        action = 0

                # MC-Dropout (Monte Carlo Dropout)
                elif cfg.backtest.selection_strategy == "ensemble_q_filter":
                    q_mean, q_std = agent.predict_ensemble(
                        state=obs,
                        training=False,
                        use_cache=cfg.backtest.use_cache,
                        cache_key=cache_key,
                        n_samples=cfg.backtest.ensemble_n_samples,
                    )
                    # Приведение типов и страховка от скаляра
                    q_mean = np.asarray(q_mean, dtype=np.float32)
                    if q_mean.ndim == 0:
                        logging.warning("predict_ensemble returned scalar q_mean; fallback to select_action(return_qvals=True, no-cache).")
                        # Важно: не используем кеш, т.к. qval_cache может содержать tuple (mean,std) от predict_ensemble
                        q_vals = agent.select_action(
                            state=obs, training=False, return_qvals=True,
                            use_cache=False, cache_key=None
                        )
                        q_mean = np.asarray(q_vals, dtype=np.float32)
                        q_std = np.zeros_like(q_mean, dtype=np.float32)
                    else:
                        q_std = np.asarray(q_std, dtype=np.float32) if np.ndim(q_std) else np.zeros_like(q_mean, dtype=np.float32)

                    advantage = q_mean - q_mean[0]
                    action = int(np.argmax(advantage))

                    # Correctly check confidence AND uncertainty
                    confidence_ok = (
                        (action == 1 and advantage[action] >= cfg.backtest.long_action_threshold) or
                        (action == 2 and advantage[action] >= cfg.backtest.short_action_threshold)
                    )

                    uncertainty_ok = q_std[action] < cfg.backtest.ensemble_max_sigma

                    if not (confidence_ok and uncertainty_ok):
                        action = 0

                else:
                    action = agent.select_action(
                        state=obs,
                        training=False,
                        return_qvals=False,
                        use_cache=cfg.backtest.use_cache,
                        cache_key=cache_key,
                    )

                obs, _, done, _, info = env.backtest_step(
                    action=action,
                    signal_dt=signal_dt,
                    ticker=ticker_name,
                    stop_loss=cfg.backtest.stop_loss,
                    take_profit=cfg.backtest.take_profit,
                    trailing_stop=cfg.backtest.trailing_stop,
                )

                if info["position_closed"]:
                    info["ticker"] = ticker_name
                    trade_log.log_trade(info, balance)
                    balance += info.get("trade_realized_pnl", 0.0)
                    result.update(signal_dt + dt.timedelta(minutes=cfg.seq.agent_session_len), info, balance)
                if done:
                    break

            open_sessions.append({"end_time": signal_dt + dt.timedelta(minutes=cfg.seq.agent_session_len)})

    agent.save_disk_cache()

    logging.info("\n[Trades Summary]:")
    trade_log.dump()

    metrics = result.finalize()
    logging.info("\n[Final Metrics]:")
    for name_result, value in metrics.items():
        logging.info(f": {name_result:>23s} = {value}")

    if cfg.backtest.plot_backtest_balance_curve:
        os.makedirs(cfg.paths.plot_dir, exist_ok=True)
        result.plot_balance(os.path.join(cfg.paths.plot_dir, "backtest_balance_curve.png"))

    return metrics


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

    # This part is a fallback for when the script is run without a config that has the 'data' dict
    if not hasattr(cfg_mod, "data"):
        from configs import alpha as cfg_mod

    run_backtest(cfg=cfg, model_path_override=model_path_arg)
