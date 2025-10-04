# backtest_continuous.py
# MODIFIED from backtest_engine.py to support continuous data backtesting.

import datetime as dt
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MasterConfig
from config import cfg as default_cfg
from test_agent import init_agent
from trading_environment import TradingEnvironment
from utils import (
    calculate_normalization_stats,
    create_signal_groups, # Kept for reference, but not used for main loop
    load_config,
    load_npz_dataset, # Kept for loading training data
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
    log_file = os.path.join(log_dir, "continuous_backtest_session.log") # New log file

    logger = logging.getLogger()

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info("[Init] Logging for continuous backtest session started")


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
                f"{int((pnl_by_day > 0).sum())} ({((pnl_by_day > 0).sum() / trade_days) * 100:.2f}%)"
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
                f"{(pnl_by_day.mean() / (std_pnl_by_day_neg + 1e-9)) * np.sqrt(len(pnl_by_day)):.2f}"
                if len(pnl_by_day) > 0
                else "0.00"
            ),
            "trades_sharpe": (f"{pnl_all.mean() / (pnl_all.std() + 1e-9):.2f}" if len(pnl_all) > 0 else "0.00"),
            "trades_sortino": (f"{pnl_all.mean() / (std_pnl_all_neg + 1e-9):.2f}" if len(pnl_all) > 0 else "0.00"),
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
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


def run_backtest(cfg: MasterConfig) -> Dict[str, Any]:
    cfg.backtest_mode = True
    setup_logging(cfg)
    set_random_seed(cfg.random_seed)

    # --- MODIFIED: Load continuous data ---
    logging.info(f"Loading continuous backtest data from {cfg.paths.backtest_data_path}")
    with np.load(cfg.paths.backtest_data_path, allow_pickle=True) as data: # allow_pickle for string array
        # Expects the NPZ from export_to_npz.py
        df = pd.DataFrame({key: data[key] for key in data.files})

    # --- NEW: Filter data for the specified ticker ---
    ticker_name = cfg.backtest.ticker_name
    if ticker_name:
        logging.info(f"Filtering data for ticker: {ticker_name}")
        df = df[df['symbol'] == ticker_name].copy()
        if df.empty:
            raise SystemExit(f"No data found for ticker {ticker_name} in the .npz file.")
    else:
        raise SystemExit("No ticker_name specified in the backtest configuration.")

    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('ts')

    logging.info(f"Loaded and filtered data with {len(df)} rows, from {df.index[0]} to {df.index[-1]}")
    market_data = df[cfg.data.data_channels].to_numpy(dtype=np.float32)

    # --- UNCHANGED: Load training data to calculate normalization stats ---
    train_raw = load_npz_dataset(
        file_path=cfg.paths.train_data_path,
        name_dataset="Train",
        plot_dir=cfg.paths.plot_dir,
        debug_max_size=cfg.debug.debug_max_size_data,
        plot_examples=0,
        plot_channel_idx=None,
        pre_signal_len=cfg.seq.pre_signal_len,
    )
    train_seqs = []
    for _, arr in train_raw:
        sel = select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels)
        if sel is not None:
            train_seqs.append(sel)
    stats = calculate_normalization_stats(
        train_seqs,
        cfg.data.data_channels,
        cfg.data.price_channels,
        cfg.data.volume_channels,
        cfg.data.other_channels,
    )

    # --- MODIFIED: Agent initialization ---
    if cfg.paths.extra_model_dir:
        # If a specific model directory is provided, use it directly
        model_folder = cfg.paths.extra_model_dir
        logging.info(f"Using specified model folder: {model_folder}")
    else:
        # Otherwise, find the latest model in the default directory
        model_base = cfg.paths.model_dir
        logging.info(f"Searching for latest model in: {model_base}")
        model_folder = os.path.join(model_base, sorted(os.listdir(model_base))[-1])

    best_path = os.path.join(model_folder, "best.pth")
    model_path = best_path if os.path.exists(best_path) else os.path.join(model_folder, "final.pth")
    logging.info(f"Loading agent from: {model_path}")
    agent = init_agent(model_path, cfg, cfg.paths.extra_cache_dir or cfg.paths.cache_dir)

    if cfg.backtest.clear_disk_cache:
        agent.clear_disk_cache()

    result = MetricsCollector()
    trade_log = TradeSummary()
    balance = cfg.market.initial_balance
    
    logging.info("\n[Starting continuous backtest...]:")

    # --- MODIFIED: Main loop with session timeout and volatility filter ---
    position_open = False
    trade_entry_step = 0
    trade_entry_price = 0.0
    trade_direction = 0 # 1 for LONG, -1 for SHORT
    
    # Get column indices for volatility calculation
    close_idx = cfg.data.data_channels.index("close")

    iterator = range(cfg.seq.full_seq_len, len(market_data))
    for i in tqdm(iterator, desc="Running Continuous Backtest"):
        session_window = market_data[i - cfg.seq.full_seq_len : i]
        current_time = df.index[i-1]
        current_price = session_window[-1][close_idx]

        action = 0 # Default to PASS

        # 1. Check for forced session timeout or if agent wants to close
        if position_open:
            if (i - trade_entry_step) >= cfg.seq.agent_session_len:
                action = 3 # Force CLOSE action
                logging.info(f"Force closing position at {current_time} due to session timeout ({cfg.seq.agent_session_len} mins).")
            else:
                # If in an open position, we still need to ask the agent if it wants to close
                temp_env = TradingEnvironment(
                    sequences=[session_window], stats=stats, render_mode=None, 
                    full_seq_len=cfg.seq.full_seq_len, num_features=cfg.seq.num_features,
                    num_actions=cfg.market.num_actions, flat_state_size=cfg.seq.flat_state_size,
                    initial_balance=balance, pre_signal_len=cfg.seq.pre_signal_len,
                    data_channels=cfg.data.data_channels, slippage=cfg.market.slippage,
                    transaction_fee=cfg.market.transaction_fee, agent_session_len=cfg.seq.agent_session_len,
                    agent_history_len=cfg.seq.agent_history_len, input_history_len=cfg.seq.input_history_len,
                    price_channels=cfg.data.price_channels, volume_channels=cfg.data.volume_channels,
                    other_channels=cfg.data.other_channels, action_history_len=cfg.seq.action_history_len,
                    inaction_penalty_ratio=cfg.market.inaction_penalty_ratio, backtest_mode=True,
                    use_risk_management=cfg.backtest.use_risk_management
                )
                # Manually set the environment state to reflect the open position
                temp_env.current_seq = temp_env.sequences[0]
                temp_env.position = trade_direction
                temp_env.entry_price = trade_entry_price
                temp_env.step_idx = i - trade_entry_step # Set the correct step index
                obs = temp_env._get_observation()

                agent_action = agent.select_action(
                    state=obs, training=False, return_qvals=False, use_cache=cfg.backtest.use_cache,
                    cache_key=(ticker_name, current_time)
                )
                if agent_action == 3:
                    action = 3 # Agent wants to close

        # 2. If not in a position, check for volatility and ask agent for an action
        elif not position_open:
            if cfg.backtest.volatility_threshold is not None:
                volatility_window = session_window[0:cfg.seq.pre_signal_len]
                close_price_start = volatility_window[0, close_idx]
                close_price_end = volatility_window[-1, close_idx]
                
                if close_price_start > 0:
                    volatility = abs(close_price_end - close_price_start) / close_price_start
                    if volatility < cfg.backtest.volatility_threshold:
                        continue # Skip if volatility is below threshold
                else:
                    continue # Skip if start price is zero

            # Volatility is high enough, or no threshold is set. Ask the agent.
            temp_env = TradingEnvironment(
                sequences=[session_window], stats=stats, render_mode=None, 
                full_seq_len=cfg.seq.full_seq_len, num_features=cfg.seq.num_features,
                num_actions=cfg.market.num_actions, flat_state_size=cfg.seq.flat_state_size,
                initial_balance=balance, pre_signal_len=cfg.seq.pre_signal_len,
                data_channels=cfg.data.data_channels, slippage=cfg.market.slippage,
                transaction_fee=cfg.market.transaction_fee, agent_session_len=cfg.seq.agent_session_len,
                agent_history_len=cfg.seq.agent_history_len, input_history_len=cfg.seq.input_history_len,
                price_channels=cfg.data.price_channels, volume_channels=cfg.data.volume_channels,
                other_channels=cfg.data.other_channels, action_history_len=cfg.seq.action_history_len,
                inaction_penalty_ratio=cfg.market.inaction_penalty_ratio, backtest_mode=True,
                use_risk_management=cfg.backtest.use_risk_management
            )
            obs, _ = temp_env.reset()
            action = agent.select_action(
                state=obs, training=False, return_qvals=False, use_cache=cfg.backtest.use_cache,
                cache_key=(ticker_name, current_time)
            )

        # 3. Process the action (Open, Close, or Hold)
        if not position_open and action in {1, 2}: # Open a new position
            position_open = True
            trade_entry_step = i
            trade_direction = 1 if action == 1 else -1
            slippage_multiplier = (1 + cfg.market.slippage) if trade_direction == 1 else (1 - cfg.market.slippage)
            trade_entry_price = current_price * slippage_multiplier
            direction_str = "LONG" if action == 1 else "SHORT"
            logging.info(f": ({direction_str}) OPEN at {trade_entry_price:.2f} on {current_time.strftime('%Y-%m-%d %H:%M')}")

        elif position_open and action == 3: # Close the current position
            slippage_multiplier = (1 - cfg.market.slippage) if trade_direction == 1 else (1 + cfg.market.slippage)
            exit_price = current_price * slippage_multiplier
            
            trade_amount = balance * cfg.backtest.position_fraction
            volume = trade_amount / trade_entry_price
            pnl = (exit_price - trade_entry_price) * volume * trade_direction
            
            # Recalculate fees for this trade
            entry_fee = trade_amount * cfg.market.transaction_fee
            exit_fee = (volume * exit_price) * cfg.market.transaction_fee
            total_fees = entry_fee + exit_fee
            net_pnl = pnl - total_fees

            info = {
                "ticker": ticker_name,
                "position_closed": True, "trade_realized_pnl": net_pnl, "total_commission": total_fees,
                "trade_amount": trade_amount, 
                "trade_price_delta": (exit_price - trade_entry_price) / trade_entry_price * trade_direction,
                "max_drawdown": 0,  # Simplified: not tracking intra-trade drawdown
                "correct_prediction": net_pnl > 0,
                "direction": "LONG" if trade_direction == 1 else "SHORT",
                "trade_dt": df.index[trade_entry_step - cfg.seq.full_seq_len].to_pydatetime(),
            }

            trade_log.log_trade(info, balance)
            balance += net_pnl
            result.update(current_time, info, balance)

            position_open = False
            trade_entry_step = 0
            trade_entry_price = 0.0
            trade_direction = 0

    agent.save_disk_cache()

    logging.info("\n[Trades Summary]:")
    trade_log.dump()

    metrics = result.finalize()
    logging.info("\n[Final Metrics]:")
    for name_result, value in metrics.items():
        logging.info(f": {name_result:>23s} = {value}")

    if cfg.backtest.plot_backtest_balance_curve:
        result.plot_balance(os.path.join(cfg.paths.plot_dir, "continuous_backtest_balance_curve.png"))

    return metrics


if __name__ == "__main__":
    # Expects a config file, e.g., python backtest_continuous.py configs/alpha.py
    run_backtest(load_config(sys.argv[1]) if len(sys.argv) > 1 else default_cfg)
