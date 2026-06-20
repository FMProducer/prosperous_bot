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

# All tickers from tickers.txt — 80% for train, 20% for OOS validation
TICKERS_FILE = PROJECT_DIR / "tickers.txt"

def load_tickers() -> List[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"{TICKERS_FILE} not found")
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

ALL_TICKERS = load_tickers()
split_idx = int(len(ALL_TICKERS) * 0.8)
TRAIN_TICKERS = ALL_TICKERS[:split_idx]
VALIDATION_TICKERS = ALL_TICKERS[split_idx:]

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

def calculate_sortino(history: list, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio from equity curve.
    Sortino = (Mean_Return - RF) / Downside_Deviation
    """
    if len(history) < 2:
        return 0.0

    # Calculate per-bar returns (percentage)
    returns = []
    for i in range(1, len(history)):
        if history[i - 1] != 0:
            ret = (history[i] - history[i - 1]) / history[i - 1]
        else:
            ret = 0.0
        returns.append(ret)

    if not returns:
        return 0.0

    returns = np.array(returns)
    mean_return = np.mean(returns)

    # Downside deviation: std of negative returns only
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        # No downside — return is infinitely good, cap at large number
        return mean_return * 1000.0 if mean_return > 0 else 0.0

    downside_std = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else np.std(negative_returns)

    if downside_std == 0:
        return mean_return * 1000.0 if mean_return > 0 else 0.0

    sortino = (mean_return - risk_free_rate) / downside_std
    return float(sortino)


def objective(trial: optuna.Trial, days: float) -> float:
    cfg = load_config()

    # Optimization space: ±30% from current config values
    # Current: trend_min_move_pct=0.7, trend_eff_threshold=0.45, net_move_block_pct=2.2, net_move_window_sec=60
    trend_params = {
        "trend_min_move_pct": trial.suggest_float("trend_min_move_pct", 0.49, 0.91, step=0.01),
        "trend_eff_threshold": trial.suggest_float("trend_eff_threshold", 0.32, 0.59, step=0.01),
        "net_move_block_pct": trial.suggest_float("net_move_block_pct", 1.54, 2.86, step=0.01),
        "net_move_window_sec": trial.suggest_int("net_move_window_sec", 42, 78, step=6),
        "trend_lookback_bars": trial.suggest_int("trend_lookback_bars", 10, 60, step=5)
    }

    scores = []
    for ticker in TRAIN_TICKERS:
        res = evaluate_ticker(cfg, ticker, days, trend_params)

        profit = res.get("profit_pct", 0.0)
        max_dd = res.get("max_dd_pct", 0.0)
        cycles = res.get("cycles", 0)
        liqs = res.get("liquidations", 0)
        tg_blocks = res.get("trend_guard_blocks", 0)
        history = res.get("history", [])

        # Hard constraints
        if liqs > 0 or max_dd > 40.0:
            scores.append(-100.0)
            continue

        # Sortino ratio from equity curve
        sortino = calculate_sortino(history) if len(history) > 1 else 0.0

        # Penalties for edge cases
        if cycles < 10:
            sortino *= 0.2  # Overly restrictive
        if tg_blocks > (cycles * 3):
            sortino *= 0.5  # Blocking too frequently

        scores.append(sortino)

    median_score = float(np.median(scores))
    trial.set_user_attr("median_score", round(median_score, 4))

    return median_score

def visualize_best_result(cfg: Dict[str, Any], best_params: Dict[str, Any], days: float):
    """Plots aggregated equity curve + prints per-ticker params table with averages."""
    print(f"\nGenerating visualization for {len(TRAIN_TICKERS)} train tickers...")

    all_histories = []
    all_logs = []
    per_ticker = {}
    per_ticker_params = {}

    # --- Per-ticker optimization ---
    param_keys = ["trend_min_move_pct", "trend_eff_threshold", "net_move_block_pct", "net_move_window_sec", "trend_lookback_bars"]

    for ticker in TRAIN_TICKERS:
        res = evaluate_ticker(cfg, ticker, days, best_params)
        if res and "history" in res and len(res["history"]) > 0:
            all_histories.append(res["history"])
            all_logs.append((ticker, res.get("rebalance_log", [])))
            per_ticker[ticker] = res
            per_ticker_params[ticker] = {k: best_params[k] for k in param_keys}

    if not all_histories:
        print("No history data available for visualization.")
        return

    # --- Determine common length ---
    min_len = min(len(h) for h in all_histories)

    # --- Build aggregated curve ---
    aggregated = []
    for i in range(min_len):
        vals = [h[i] for h in all_histories if i < len(h)]
        aggregated.append(sum(vals) / len(vals))

    # --- Aggregate rebalance points ---
    rebalance_counts = [0] * min_len
    for ticker, logs in all_logs:
        for log in logs:
            step = log.get("step", 0)
            if 0 <= step < min_len:
                rebalance_counts[step] += 1

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(aggregated, label=f"Avg TPV across {len(all_histories)} tickers", color="blue", linewidth=2)

    reb_x = [i for i in range(min_len) if rebalance_counts[i] > 0]
    reb_y = [aggregated[i] for i in reb_x]
    if reb_x:
        ax1.scatter(reb_x, reb_y, color="green", marker="^", s=30, alpha=0.6, label=f"Rebalance events (total {sum(rebalance_counts)})")

    ax1.set_title(f"Trend Guard Optimization — Aggregated TPV ({len(TRAIN_TICKERS)} tickers, {days}d)")
    ax1.set_ylabel("Avg TPV (USDT)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.bar(range(min_len), rebalance_counts, color="green", alpha=0.5, label="Rebalance count per bar")
    ax2.set_xlabel("Time (Bars)")
    ax2.set_ylabel("Rebalance count")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = PROJECT_DIR / "trend_guard_optimization_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Visualization saved to {plot_path}")

    # --- Print per-ticker params table ---
    print(f"\n{'Ticker':<18} {'t_min_move':>10} {'t_eff_thr':>10} {'nmg_pct':>9} {'nmg_win':>8} {'lookback':>8} {'Profit':>9} {'MaxDD':>7} {'Cycles':>7} {'TG Blk':>6}")
    print("-" * 115)

    sums = {k: 0.0 for k in param_keys}
    counts = 0

    for ticker in TRAIN_TICKERS:
        r = per_ticker.get(ticker, {})
        p = per_ticker_params.get(ticker, {})
        profit = r.get("profit_pct", 0.0)
        max_dd = r.get("max_dd_pct", 0.0)
        cycles = r.get("cycles", 0)
        tg_blocks = r.get("trend_guard_blocks", 0)

        tmm = p.get("trend_min_move_pct", 0)
        tet = p.get("trend_eff_threshold", 0)
        nmg_pct = p.get("net_move_block_pct", 0)
        nmg_win = p.get("net_move_window_sec", 0)
        lb = p.get("trend_lookback_bars", 0)

        print(f"{ticker:<18} {tmm:>10.2f} {tet:>10.2f} {nmg_pct:>9.2f} {nmg_win:>8} {lb:>8} {profit:>+8.2f}% {max_dd:>6.2f}% {cycles:>6} {tg_blocks:>5}")

        for k in param_keys:
            sums[k] += p.get(k, 0)
        counts += 1

    # --- Averages row ---
    if counts > 0:
        avgs = {k: sums[k] / counts for k in param_keys}
        avg_profit = sum(r.get("profit_pct", 0) for r in per_ticker.values()) / counts
        avg_dd = sum(r.get("max_dd_pct", 0) for r in per_ticker.values()) / counts
        avg_cycles = sum(r.get("cycles", 0) for r in per_ticker.values()) / counts
        avg_tg = sum(r.get("trend_guard_blocks", 0) for r in per_ticker.values()) / counts

        print("-" * 115)
        print(f"{'AVERAGE':<18} {avgs['trend_min_move_pct']:>10.2f} {avgs['trend_eff_threshold']:>10.2f} {avgs['net_move_block_pct']:>9.2f} {avgs['net_move_window_sec']:>8.0f} {avgs['trend_lookback_bars']:>8.0f} {avg_profit:>+8.2f}% {avg_dd:>6.2f}% {avg_cycles:>6.0f} {avg_tg:>5.0f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--days", type=float, default=0.125, help="Backtest period in days (default: 0.125 = 3 hours)")
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

    # --- Save results to JSON ---
    output = {
        "ticker": "ALL_FROM_TICKERS_TXT",
        "days": args.days,
        "trials": len(study.trials),
        "elapsed_sec": None,
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": best.params,
        "best_profit_pct": best.user_attrs.get("profit_pct"),
        "best_max_dd_pct": best.user_attrs.get("max_dd_pct"),
        "best_liquidations": best.user_attrs.get("liquidations"),
        "train_tickers": TRAIN_TICKERS,
        "validation_tickers": VALIDATION_TICKERS,
    }
    output_path = PROJECT_DIR / "best_trend_guard.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Cleanup temporary patch
    patched_path = PROJECT_DIR / "_patched_backtest.py"
    if patched_path.exists():
        patched_path.unlink()

if __name__ == "__main__":
    main()
