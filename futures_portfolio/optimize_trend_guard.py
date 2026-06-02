"""
Optuna-оптимизация Trend Guard параметров для PORTALUSDT.

Trend Guard в main.py:
  - trend_min_move_pct: минимум движения для распознавания тренда (0.5% сейчас)
  - trend_eff_threshold: эффективность тренда (0.85 сейчас) — доля движения в одну сторону
  - net_move_block_pct: порог NMG для блокировки (1.5% сейчас)
  - net_move_window_sec: окно NMG (30с сейчас)

Проблема: при плавном тренде trend_eff < 0.85 → guard не срабатывает.
Цель: найти пороги, при которых guard блокирует ребаланс при тренде,
       но не блокирует при флэте.

Запуск:
  python optimize_trend_guard.py --ticker PORTALUSDT --n-trials 200
"""

import asyncio
import json
import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path
from decimal import Decimal

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR = PROJECT_DIR / "data"
STORAGE_PATH = PROJECT_DIR / "optuna_trend_guard.db"

sys.path.insert(0, str(PROJECT_DIR))
from backtest_rebalance import run_backtest

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("OptunaTrend")
warnings.filterwarnings("ignore")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_backtest_with_trend_guard(cfg, ticker, days, trend_params):
    """
    Run backtest with Trend Guard simulated in config.
    We patch the safety_guards section with new trend guard params.
    """
    # Deep copy to avoid mutation
    import copy
    patched = copy.deepcopy(cfg)
    
    # Patch safety_guards
    guards = patched.get("portfolios", [{}])[0].get("safety_guards", {})
    guards["net_move_block_pct"] = trend_params["net_move_block_pct"]
    guards["net_move_window_sec"] = trend_params["net_move_window_sec"]
    # Add new params for trend guard (used by main.py)
    guards["trend_min_move_pct"] = trend_params["trend_min_move_pct"]
    guards["trend_eff_threshold"] = trend_params["trend_eff_threshold"]
    
    # Also save at top level for backtest access
    patched["trend_guard"] = {
        "min_move_pct": trend_params["trend_min_move_pct"],
        "eff_threshold": trend_params["trend_eff_threshold"],
        "net_move_block_pct": trend_params["net_move_block_pct"],
        "net_move_window_sec": trend_params["net_move_window_sec"],
    }
    
    tmp_config = PROJECT_DIR / f"_optuna_trend_tmp_{ticker}.json"
    with open(tmp_config, "w", encoding="utf-8") as f:
        json.dump(patched, f, indent=2)
    try:
        result = asyncio.run(
            run_backtest(
                config_path=str(tmp_config),
                data_dir=str(DATA_DIR),
                live_mode=False,
                ticker_override=ticker,
                days=days,
                quiet=True,
            )
        )
        return result
    finally:
        if tmp_config.exists():
            tmp_config.unlink()


def objective(trial: optuna.Trial, ticker: str, days: float) -> float:
    """One trial = one Trend Guard param combination on one ticker."""
    cfg = load_config()
    
    # --- Search space ---
    trend_min_move_pct = trial.suggest_float("trend_min_move_pct", 0.1, 2.0, step=0.1)
    trend_eff_threshold = trial.suggest_float("trend_eff_threshold", 0.3, 0.95, step=0.05)
    net_move_block_pct = trial.suggest_float("net_move_block_pct", 0.3, 3.0, step=0.1)
    net_move_window_sec = trial.suggest_int("net_move_window_sec", 10, 120, step=5)
    
    trend_params = {
        "trend_min_move_pct": trend_min_move_pct,
        "trend_eff_threshold": trend_eff_threshold,
        "net_move_block_pct": net_move_block_pct,
        "net_move_window_sec": net_move_window_sec,
    }
    
    trial.set_user_attr("trend_params", json.dumps(trend_params))
    
    # --- Run backtest ---
    try:
        result = run_backtest_with_trend_guard(cfg, ticker, days, trend_params)
    except Exception as e:
        logger.warning(f"Backtest exception: {e}")
        raise optuna.TrialPruned()
    
    if result is None:
        raise optuna.TrialPruned()
    
    profit = result.get("profit_pct", 0.0)
    max_dd = result.get("max_dd_pct", 0.0)
    cycles = result.get("cycles", 0)
    liqs = result.get("liquidations", 0)
    
    trial.set_user_attr("profit_pct", round(profit, 4))
    trial.set_user_attr("max_dd_pct", round(max_dd, 2))
    trial.set_user_attr("cycles", cycles)
    trial.set_user_attr("liquidations", liqs)
    
    # --- Pruning ---
    trial.report(profit, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # --- Penalties ---
    # Liquidations are catastrophic
    if liqs > 0:
        profit *= 0.1
    
    # Max DD > 30% is dangerous
    if max_dd > 30:
        profit *= 0.3
    
    # Negative profit but low DD — still penalize
    if profit < 0 and max_dd < 10:
        profit *= 0.5  # boring but losing
    
    # Profit/DD ratio (risk-adjusted)
    if max_dd > 1:
        score = profit / max_dd  # higher = better risk-adjusted
    else:
        score = profit
    
    # Favor: high profit, low DD, no liquidations
    final_score = profit * 0.7 + score * 10 * 0.3
    
    trial.set_user_attr("final_score", round(final_score, 4))
    
    return round(final_score, 4)


def main():
    parser = argparse.ArgumentParser(description="Optuna Trend Guard Optimization")
    parser.add_argument("--ticker", type=str, default="PORTALUSDT")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--days", type=float, default=0.5, help="Backtest period in days")
    parser.add_argument("--timeout", type=int, default=None, help="Total time limit (sec)")
    parser.add_argument("--storage", type=str, default=str(STORAGE_PATH))
    parser.add_argument("--study-name", type=str, default="trend_guard_v1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger("Backtest").setLevel(logging.INFO)
    
    cfg = load_config()
    
    print("=" * 60)
    print("  Optuna Trend Guard Optimization")
    print(f"  Ticker:   {args.ticker}")
    print(f"  Days:     {args.days} ({args.days * 24:.1f}h)")
    print(f"  Trials:   {args.n_trials}")
    print(f"  Storage:  {args.storage}")
    print("=" * 60)
    
    storage_url = f"sqlite:///{args.storage}"
    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )
    
    start_time = time.time()
    
    study.optimize(
        lambda trial: objective(trial, args.ticker, args.days),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # --- Results ---
    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    best = study.best_trial
    
    print(f"\n  Лучший trial #{best.number}")
    print(f"  Profit:     {best.user_attrs.get('profit_pct', '?')}%")
    print(f"  Max DD:     {best.user_attrs.get('max_dd_pct', '?')}%")
    print(f"  Cycles:     {best.user_attrs.get('cycles', '?')}")
    print(f"  Liquidations: {best.user_attrs.get('liquidations', '?')}")
    print(f"  Score:      {best.user_attrs.get('final_score', '?')}")
    
    print(f"\n  Trend Guard параметры:")
    for name, val in best.params.items():
        print(f"    {name:<30s} = {val}")
    
    # Current vs Best comparison
    current = {
        "trend_min_move_pct": 0.5,
        "trend_eff_threshold": 0.85,
        "net_move_block_pct": 1.5,
        "net_move_window_sec": 30,
    }
    print(f"\n  Сравнение с текущими:")
    for name in best.params:
        cur = current.get(name, "?")
        new = best.params[name]
        changed = "← CHANGE" if cur != new else ""
        print(f"    {name:<30s}  {cur} → {new}  {changed}")
    
    # Top-5
    print(f"\n  Top-5 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:5]
    for rank, t in enumerate(top5, 1):
        prof = t.user_attrs.get("profit_pct", "?")
        dd = t.user_attrs.get("max_dd_pct", "?")
        liq = t.user_attrs.get("liquidations", "?")
        print(f"    #{rank} Trial {t.number:>3d}: profit={prof:>8}%  max_dd={dd:>5}%  liq={liq}")
    
    # Save
    output = {
        "ticker": args.ticker,
        "days": args.days,
        "trials": len(study.trials),
        "elapsed_sec": round(elapsed, 1),
        "best_params": best.params,
        "best_profit_pct": best.user_attrs.get("profit_pct"),
        "best_max_dd_pct": best.user_attrs.get("max_dd_pct"),
        "best_liquidations": best.user_attrs.get("liquidations"),
    }
    output_path = PROJECT_DIR / "best_trend_guard.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Сохранено: {output_path}")
    print(f"  Время: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
