"""
Multi-Ticker Optuna Optimization for Trend Guard with Runtime Patching.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR = PROJECT_DIR / "data"
STORAGE_PATH = PROJECT_DIR / "optuna_trend_guard.db"

# Ticker splits for Train and Out-of-Sample (OOS) validation
TRAIN_TICKERS = ["PORTALUSDT", "NEARUSDT", "STGUSDT", "BRUSDT"]
VALIDATION_TICKERS = ["1000BONKUSDT", "SUIUSDT", "TAOUSDT"]

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("OptunaTrend")

def generate_patched_backtester() -> str:
    """
    Dynamically patches backtest_rebalance.py to read window parameters from config
    and return history logs for visualization, bypassing the need to modify the original file.
    """
    original_path = PROJECT_DIR / "backtest_rebalance.py"
    patched_path = PROJECT_DIR / "_patched_backtest.py"

    with open(original_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Inject config lookups for hardcoded local variables
    code = re.sub(
        r"tg_nmg_bars = max\(1, tg_nmg_window // 60\)",
        r"tg_nmg_bars = max(1, trend_guard_cfg.get('net_move_window_sec', 60) // 60)",
        code
    )
    code = re.sub(
        r"tg_trend_bars = 30",
        r"tg_trend_bars = trend_guard_cfg.get('lookback_bars', 30)",
        code
    )

    # Expose history and logs in the return dictionary
    code = re.sub(
        r"\"sortino_ratio\": float\(sortino_ratio\)",
        r'"sortino_ratio": float(sortino_ratio),\n            "history": state.history,\n            "rebalance_log": state.rebalance_log',
        code
    )

    with open(patched_path, "w", encoding="utf-8") as f:
        f.write(code)

    return "_patched_backtest"

# Patch and import dynamically
patched_module_name = generate_patched_backtester()
sys.path.insert(0, str(PROJECT_DIR))
patched_backtest = __import__(patched_module_name)

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_ticker(cfg: Dict[str, Any], ticker: str, days: float, trend_params: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    patched_cfg = copy.deepcopy(cfg)

    guards = patched_cfg.get("portfolios", [{}])[0].get("safety_guards", {})
    guards.update({
        "net_move_block_pct": trend_params["net_move_block_pct"],
        "net_move_window_sec": trend_params["net_move_window_sec"],
        "trend_min_move_pct": trend_params["trend_min_move_pct"],
        "trend_eff_threshold": trend_params["trend_eff_threshold"]
    })

    patched_cfg["trend_guard"] = {
        "min_move_pct": trend_params["trend_min_move_pct"],
        "eff_threshold": trend_params["trend_eff_threshold"],
        "net_move_block_pct": trend_params["net_move_block_pct"],
        "net_move_window_sec": trend_params["net_move_window_sec"],
        "lookback_bars": trend_params["trend_lookback_bars"]
    }

    tmp_config = PROJECT_DIR / f"_tmp_cfg_{ticker}.json"
    with open(tmp_config, "w", encoding="utf-8") as f:
        json.dump(patched_cfg, f)

    try:
        result = asyncio.run(
            patched_backtest.run_backtest(
                config_path=str(tmp_config),
                data_dir=str(DATA_DIR),
                live_mode=False,
                ticker_override=ticker,
                days=days,
                quiet=True,
            )
        )
        return result or {}
    finally:
        if tmp_config.exists():
            tmp_config.unlink()

def objective(trial: optuna.Trial, days: float) -> float:
    cfg = load_config()

    # Optimization space
    trend_params = {
        "trend_min_move_pct": trial.suggest_float("trend_min_move_pct", 0.2, 1.5, step=0.1),
        "trend_eff_threshold": trial.suggest_float("trend_eff_threshold", 0.3, 0.85, step=0.05),
        "net_move_block_pct": trial.suggest_float("net_move_block_pct", 0.5, 3.0, step=0.1),
        "net_move_window_sec": trial.suggest_int("net_move_window_sec", 30, 180, step=30),
        "trend_lookback_bars": trial.suggest_int("trend_lookback_bars", 10, 60, step=10)
    }

    scores = []
    for ticker in TRAIN_TICKERS:
        res = evaluate_ticker(cfg, ticker, days, trend_params)

        profit = res.get("profit_pct", 0.0)
        max_dd = res.get("max_dd_pct", 0.0)
        cycles = res.get("cycles", 0)
        liqs = res.get("liquidations", 0)
        tg_blocks = res.get("trend_guard_blocks", 0)

        if liqs > 0 or max_dd > 40.0:
            scores.append(-100.0)
            continue

        # Stabilized Sharpe-like calculation
        score = profit / (max_dd + 1.0)

        # Penalties for edge cases
        if cycles < 10:
            score *= 0.2  # Overly restrictive
        if tg_blocks > (cycles * 3):
            score *= 0.5  # Blocking too frequently

        scores.append(score)

    median_score = float(np.median(scores))
    trial.set_user_attr("median_score", round(median_score, 4))

    return median_score

def visualize_best_result(cfg: Dict[str, Any], best_params: Dict[str, Any], days: float):
    """Plots the equity curve and rebalance points for the first train ticker."""
    ticker = TRAIN_TICKERS[0]
    print(f"\nGenerating visualization for {ticker}...")
    res = evaluate_ticker(cfg, ticker, days, best_params)

    if not res or "history" not in res:
        print("No history data available for visualization.")
        return

    history = res["history"]
    logs = res["rebalance_log"]

    plt.figure(figsize=(14, 7))
    plt.plot(history, label="Total Equity (TPV)", color="blue", linewidth=1.5)

    # Scatter rebalance points
    if logs:
        x_vals = [log["step"] for log in logs]
        y_vals = [log["tpv"] for log in logs]
        plt.scatter(x_vals, y_vals, color="green", marker="^", s=50, label="Rebalance Executed")

    plt.title(f"Best Trend Guard Params Simulation: {ticker} (Profit: {res.get('profit_pct', 0):.2f}%)")
    plt.xlabel("Time (Bars)")
    plt.ylabel("TPV (USDT)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = PROJECT_DIR / "trend_guard_optimization_plot.png"
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--days", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_config()

    print("=" * 60)
    print("Multi-Ticker Trend Guard Optimization")
    print(f"Train: {TRAIN_TICKERS}")
    print(f"OOS Validation: {VALIDATION_TICKERS}")
    print("=" * 60)

    study = optuna.create_study(
        study_name="trend_guard_v2",
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner()
    )

    study.optimize(lambda t: objective(t, args.days), n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Trial #{best.number} | Median Score: {best.value:.4f}")
    print("\nOptimal Parameters:")
    for k, v in best.params.items():
        print(f"  {k:<25} = {v}")

    print("\n--- OOS VALIDATION ---")
    for ticker in VALIDATION_TICKERS:
        res = evaluate_ticker(cfg, ticker, args.days, best.params)
        prof = res.get("profit_pct", 0.0)
        dd = res.get("max_dd_pct", 0.0)
        cycles = res.get("cycles", 0)
        blocks = res.get("trend_guard_blocks", 0)
        print(f"{ticker:<15} | Profit: {prof:>+6.2f}% | MaxDD: {dd:>5.2f}% | Cycles: {cycles:>4} | TG Blocks: {blocks}")

    visualize_best_result(cfg, best.params, args.days)

    # Cleanup temporary patch
    patched_path = PROJECT_DIR / "_patched_backtest.py"
    if patched_path.exists():
        patched_path.unlink()

if __name__ == "__main__":
    main()
