# backtest_engine.py

import datetime as dt
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import MasterConfig
from config import cfg as default_cfg
from test_agent import init_agent
from trading_environment import TradingEnvironment
from utils import (
    calculate_normalization_stats,
    create_signal_groups,
    load_config,
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

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file):
            return

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler(),
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
    setup_logging(cfg)
    set_random_seed(cfg.random_seed)

    backtest_raw = load_npz_dataset(
        file_path=cfg.paths.backtest_data_path,
        name_dataset="Backtest",
        plot_dir=cfg.paths.plot_dir,
        debug_max_size=cfg.debug.debug_max_size_data,
        plot_examples=cfg.data.plot_examples,
        plot_channel_idx=cfg.data.plot_channel_idx,
        pre_signal_len=cfg.seq.pre_signal_len,
    )

    grouped_backtest_data = create_signal_groups(backtest_raw)
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

    model_base = cfg.paths.extra_model_dir or cfg.paths.model_dir
    model_folder = os.path.join(model_base, sorted(os.listdir(model_base))[-1])

    best_path = os.path.join(model_folder, "best.pth")
    model_path = best_path if os.path.exists(best_path) else os.path.join(model_folder, "final.pth")

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
                    confidence = adv[action]

                    pass_adv = get_pass_advantage(action, confidence, cfg)
                    if pass_adv:
                        logging.info(
                            f": REJECTED {['LONG', 'SHORT', 'CLOSE'][action-1]}, "
                            f"confidence={confidence:.3f} < threshold={thresholds[action-1]}"
                        )
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
                    advantage = q_mean - q_mean[0]
                    action = int(np.argmax(advantage))
                    confidence = advantage[action]
                    uncertainty = q_std[action]

                    pass_adv = get_pass_advantage(action, confidence, cfg)
                    pass_uncertainty = uncertainty >= cfg.backtest.ensemble_max_sigma
                    if pass_adv and pass_uncertainty:
                        logging.info(
                            f": REJECTED {['LONG', 'SHORT', 'CLOSE'][action-1]}, "
                            f"confidence={confidence:.3f} < threshold={thresholds[action-1]}, "
                            f"uncertainty={uncertainty:.3f} > max_sigma_threshold={cfg.backtest.ensemble_max_sigma}"
                        )
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
        result.plot_balance(os.path.join(cfg.paths.plot_dir, "backtest_balance_curve.png"))

    return metrics


if __name__ == "__main__":
    run_backtest(load_config(sys.argv[1]) if len(sys.argv) > 1 else default_cfg)
