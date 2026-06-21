"""
Multi-Ticker Optuna Optimization for Core Strategy Parameters.
Optimizes Rebalance Thresholds, Trailing Stop, Leverage, and Virtual Leg Share.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from backtest_rebalance import run_backtest

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR = PROJECT_DIR / "data"

TICKERS_FILE = PROJECT_DIR / "tickers.txt"

def load_tickers() -> List[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"{TICKERS_FILE} not found")
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

ALL_TICKERS = load_tickers()
VALIDATION_TICKERS = ALL_TICKERS[:10]
TRAIN_TICKERS = ALL_TICKERS[10:]

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("OptunaConfig")

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_ticker(cfg: Dict[str, Any], ticker: str, days: float, params: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    patched_cfg = copy.deepcopy(cfg)

    # 1. Update Core Thresholds
    patched_cfg["portfolios"][0]["rebalance_threshold_surplus"] = params["rebalance_threshold_surplus"]
    patched_cfg["portfolios"][0]["rebalance_threshold_deficit"] = params["rebalance_threshold_deficit"]

    # Remove individual ticker overrides to enforce global parameter testing
    if "ticker_thresholds" in patched_cfg["portfolios"][0]:
        patched_cfg["portfolios"][0]["ticker_thresholds"] = {}

    # 2. Update Trailing Stop
    patched_cfg["equity_trailing_stop_activation_pct"] = params["equity_trailing_stop_activation_pct"]
    patched_cfg["equity_trailing_stop_pct"] = params["equity_trailing_stop_pct"]

    # 3. Dynamic Share Balancing & Leverage
    v_share = params["virtual_share"]
    base_share = round((1.0 - v_share) / 2.0, 4)
    lev = params["leverage"]

    targets = patched_cfg["portfolios"][0]["targets"]
    targets["VIRTUAL"]["share"] = v_share
    targets["BASE_LONG"]["share"] = base_share
    targets["BASE_LONG"]["leverage"] = lev
    targets["BASE_SHORT"]["share"] = base_share
    targets["BASE_SHORT"]["leverage"] = lev

    # Thread-safe temporary config creation
    fd, tmp_config_path = tempfile.mkstemp(suffix=".json", prefix=f"tmp_core_cfg_{ticker}_", dir=str(PROJECT_DIR))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(patched_cfg, f)

    try:
        result = asyncio.run(
            run_backtest(
                config_path=tmp_config_path,
                data_dir=str(DATA_DIR),
                live_mode=False,
                ticker_override=ticker,
                days=days,
                quiet=True,
            )
        )
        return result or {}
    finally:
        try:
            os.remove(tmp_config_path)
        except OSError:
            pass

def calculate_sortino(returns_pct: float, max_dd_pct: float) -> float:
    """Simplified heuristic score matching previous robust logic."""
    if max_dd_pct <= 0:
        return returns_pct * 10.0
    return returns_pct / (max_dd_pct / 10.0 + 1.0)

def objective(trial: optuna.Trial, days: float) -> float:
    cfg = load_config()

    # Search Space Definition
    params = {
        "rebalance_threshold_surplus": trial.suggest_float("rebalance_threshold_surplus", 0.01, 0.01, step=0.01),
        "rebalance_threshold_deficit": trial.suggest_float("rebalance_threshold_deficit", 0.01, 0.09, step=0.01),
        "equity_trailing_stop_activation_pct": trial.suggest_float("equity_trailing_stop_activation_pct", 5.0, 15.0, step=1.0),
        "virtual_share": trial.suggest_float("virtual_share", 0.0, 0.0, step=0.05),
        "leverage": trial.suggest_int("leverage", 7, 7, step=1)
    }

    # Dependent variable constraint: Trailing stop MUST be less than activation
    max_ts = min(params["equity_trailing_stop_activation_pct"] - 1.0, 8.0)
    params["equity_trailing_stop_pct"] = trial.suggest_float("equity_trailing_stop_pct", 0.0001, max_ts, step=0.0001)

    scores = []
    for ticker in TRAIN_TICKERS:
        res = evaluate_ticker(cfg, ticker, days, params)

        profit = res.get("profit_pct", 0.0)
        max_dd = res.get("max_dd_pct", 0.0)
        cycles = res.get("cycles", 0)
        liqs = res.get("liquidations", 0)

        # Hard constraints
        if liqs > 0 or max_dd > 45.0:
            scores.append(-100.0)
            continue

        score = calculate_sortino(profit, max_dd)

        # Activity penalty
        if cycles < 5:
            score *= 0.1

        scores.append(score)

    median_score = float(np.median(scores))
    trial.set_user_attr("median_score", round(median_score, 4))

    return median_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--days", type=float, default=0.5, help="Backtest period in days (default: 0.5 = 12 hours)")
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Ticker Core Configuration Optimization")
    print("=" * 60)

    study = optuna.create_study(
        study_name="core_config_v1",
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner()
    )

    study.optimize(lambda t: objective(t, args.days), n_trials=args.n_trials, show_progress_bar=True, n_jobs=1)

    best = study.best_trial
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Trial #{best.number} | Median Score: {best.value:.4f}")
    print("\nOptimal Parameters:")

    v_share = best.params['virtual_share']
    base_share = round((1.0 - v_share) / 2.0, 4)
    print(f"  BASE_LONG/SHORT Share       = {base_share}")
    for k, v in best.params.items():
        print(f"  {k:<35} = {v}")

    # Output to file
    output_path = PROJECT_DIR / "best_core_config.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
