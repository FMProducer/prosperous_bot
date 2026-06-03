"""
Двухэтапная Optuna-оптимизация Take-Profit и Trailing Stop.

Этап 1: Совместная оптимизация (activation + TS) — находим оптимальную пару.
         activation: 1-20%, TS: 0.5-10%
Этап 2: Фиксируем найденный activation, доужаем TS в узком диапазоне ±2% от найденного.

Запуск:
  python optimize_takeprofit_stop.py
  python optimize_takeprofit_stop.py --n-trials-1 100 --n-trials-2 50
  python optimize_takeprofit_stop.py --study-name tp_v1 --days 2
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

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR = PROJECT_DIR / "data"

sys.path.insert(0, str(PROJECT_DIR))
from backtest_rebalance import run_backtest

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("OptunaTP")
warnings.filterwarnings("ignore")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tickers(cfg):
    tickers_file = PROJECT_DIR / "tickers.txt"
    if tickers_file.exists():
        with open(tickers_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return cfg.get("tickers", ["ALGOUSDT"])


def get_timeout_sec(cfg):
    return int(cfg.get("equity_trailing_stop_timeout_sec", 60))


def run_backtest_for_ticker(config, ticker, days):
    tmp_config = PROJECT_DIR / f"_optuna_tmp_{ticker}.json"
    with open(tmp_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
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


def run_all_tickers(cfg, tickers, days):
    per_ticker = {}
    errors = 0
    for ticker in tickers:
        try:
            result = run_backtest_for_ticker(cfg, ticker, days)
        except Exception as e:
            logger.warning(f"  {ticker}: exception — {e}")
            errors += 1
            continue
        if result is None:
            errors += 1
            continue
        per_ticker[ticker] = {
            "profit": result.get("profit_pct", 0.0),
            "max_dd": result.get("max_dd_pct", 0.0),
            "trailing_stop": result.get("trailing_stop_triggered", False),
            "cycles": result.get("cycles", 0),
        }
    return per_ticker, errors


def compute_metric(per_ticker, max_dd_limit):
    if not per_ticker:
        return 0.0, {}
    n = len(per_ticker)
    avg_profit = sum(v["profit"] for v in per_ticker.values()) / n
    avg_dd = sum(v["max_dd"] for v in per_ticker.values()) / n
    max_dd = max(v["max_dd"] for v in per_ticker.values())
    ts_count = sum(1 for v in per_ticker.values() if v["trailing_stop"])
    profitable = sum(1 for v in per_ticker.values() if v["profit"] > 0)
    pf = (avg_profit / avg_dd) if avg_dd > 0.1 else avg_profit / 0.1

    # Штраф за превышение DD
    if max_dd > max_dd_limit:
        avg_profit *= 0.1
    # Штраф за слишком частый TS
    if ts_count / n > 0.3:
        avg_profit *= 0.85
    # Штраф за большинство убыточных
    if profitable < n / 2:
        avg_profit *= 0.5
    if pf < 1.0:
        avg_profit *= 0.5

    stats = {
        "avg_profit_pct": round(avg_profit, 4),
        "avg_max_dd_pct": round(avg_dd, 2),
        "max_dd_pct": round(max_dd, 2),
        "trailing_stop_count": ts_count,
        "ts_pct_of_tickers": round(ts_count / n * 100, 1),
        "profitable": f"{profitable}/{n}",
        "profit_factor": round(pf, 3),
    }
    return round(avg_profit, 4), stats


def print_per_ticker(per_ticker, activation, ts_pct):
    if per_ticker:
        print(f"\n  Per-ticker (activation={activation}%, TS={ts_pct}%):")
        for ticker in sorted(per_ticker.keys()):
            v = per_ticker[ticker]
            ts = "TS" if v["trailing_stop"] else "  "
            print(f"    {ticker:<14s} profit={v['profit']:>+8.3f}%  dd={v['max_dd']:>6.2f}%  "
                  f"cyc={v['cycles']:>3d}  {ts}")


# ============================================================
# ЭТАП 1: Совместная оптимизация activation + TS
# ============================================================
def objective_phase1(trial: optuna.Trial, tickers, days, timeout_sec, max_dd_limit):
    cfg = load_config()

    activation = trial.suggest_float("activation_pct", 1.0, 20.0, step=0.5)
    ts_pct = trial.suggest_float("trailing_stop_pct", 0.5, 10.0, step=0.5)

    # Constraint: TS > activation бессмысленно (стоп шире чем тейкпрофит)
    # Но TS — это % от ATH, а activation — % от initial, это разные базы
    # Просто разрешаем все комбинации, Optuna сама разберётся

    cfg["equity_trailing_stop_pct"] = ts_pct
    cfg["equity_trailing_stop_activation_pct"] = activation
    cfg["equity_trailing_stop_timeout_sec"] = timeout_sec
    cfg["max_drawdown_limit"] = max_dd_limit

    per_ticker, errors = run_all_tickers(cfg, tickers, days)
    if not per_ticker:
        raise optuna.TrialPruned()

    metric, stats = compute_metric(per_ticker, max_dd_limit)
    trial.set_user_attr("activation_pct", activation)
    trial.set_user_attr("trailing_stop_pct", ts_pct)
    trial.set_user_attr("errors", errors)
    for k, v in stats.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("per_ticker", json.dumps(per_ticker))

    trial.report(metric, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return metric


# ============================================================
# ЭТАП 2: Доужение TS при фиксированном activation
# ============================================================
def objective_phase2(trial: optuna.Trial, tickers, days, timeout_sec,
                     best_activation, ts_center, max_dd_limit):
    cfg = load_config()

    # Узкий диапазон ±2% от найденного TS, но не ниже 0.1%
    ts_low = max(0.1, ts_center - 2.0)
    ts_high = ts_center + 2.0
    ts_pct = trial.suggest_float("trailing_stop_pct", ts_low, ts_high, step=0.1)

    cfg["equity_trailing_stop_pct"] = ts_pct
    cfg["equity_trailing_stop_activation_pct"] = best_activation
    cfg["equity_trailing_stop_timeout_sec"] = timeout_sec
    cfg["max_drawdown_limit"] = max_dd_limit

    per_ticker, errors = run_all_tickers(cfg, tickers, days)
    if not per_ticker:
        raise optuna.TrialPruned()

    metric, stats = compute_metric(per_ticker, max_dd_limit)
    trial.set_user_attr("trailing_stop_pct", ts_pct)
    trial.set_user_attr("activation_fixed", best_activation)
    trial.set_user_attr("errors", errors)
    for k, v in stats.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("per_ticker", json.dumps(per_ticker))

    trial.report(metric, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return metric


def main():
    parser = argparse.ArgumentParser(
        description="Двухэтапная оптимизация: Take-Profit + Trailing Stop")
    parser.add_argument("--n-trials-1", type=int, default=80,
                        help="Trials этап 1 (activation + TS)")
    parser.add_argument("--n-trials-2", type=int, default=40,
                        help="Trials этап 2 (TS refine)")
    parser.add_argument("--study-name", type=str, default="takeprofit_stop")
    parser.add_argument("--days", type=float, default=None)
    parser.add_argument("--max-dd", type=float, default=30.0)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("Backtest").setLevel(logging.INFO)

    cfg = load_config()
    tickers = get_tickers(cfg)
    days = args.days if args.days else float(cfg.get("backtest_period_days", 2.0))
    timeout_sec = get_timeout_sec(cfg)
    max_dd_limit = args.max_dd

    storage_path = args.storage or str(PROJECT_DIR / "optuna_takeprofit.db")
    storage_url = f"sqlite:///{storage_path}"

    print("=" * 70)
    print("  ДВУХЭТАПНАЯ ОПТИМИЗАЦИЯ: Take-Profit + Trailing Stop")
    print("=" * 70)
    print(f"  Тикеров:       {len(tickers)} ({', '.join(tickers[:5])}...)")
    print(f"  Дней:          {days} ({days * 24:.1f}h)")
    print(f"  max_dd_limit:  {max_dd_limit}% (fixed)")
    print(f"  Timeout:       {timeout_sec}s")
    print(f"  Trials:        этап1={args.n_trials_1}, этап2={args.n_trials_2}")
    print()

    # ================================================================
    # ЭТАП 1
    # ================================================================
    print("=" * 70)
    print("  ЭТАП 1: Совместная оптимизация activation + TS")
    print("          activation: 1-20%, TS: 0.5-10%")
    print("=" * 70)

    study1 = optuna.create_study(
        study_name=f"{args.study_name}_phase1",
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        storage=storage_url,
        load_if_exists=True,
    )

    start1 = time.time()
    study1.optimize(
        lambda trial: objective_phase1(trial, tickers, days, timeout_sec, max_dd_limit),
        n_trials=args.n_trials_1,
        show_progress_bar=True,
    )
    elapsed1 = time.time() - start1

    best1 = study1.best_trial
    best_activation = best1.params["activation_pct"]
    best_ts_coarse = best1.params["trailing_stop_pct"]

    print(f"\n  РЕЗУЛЬТАТ ЭТАПА 1 ({elapsed1:.0f}s):")
    print(f"  ✓ activation_pct = {best_activation}%  (take-profit)")
    print(f"  ✓ trailing_stop_pct = {best_ts_coarse}%  (coarse stop)")
    print(f"    Avg Profit:    {best1.user_attrs.get('avg_profit_pct', '?')}%")
    print(f"    Avg Max DD:    {best1.user_attrs.get('avg_max_dd_pct', '?')}%")
    print(f"    Profit Factor: {best1.user_attrs.get('profit_factor', '?')}")
    print(f"    Profitable:    {best1.user_attrs.get('profitable', '?')}")
    print(f"    TS triggers:   {best1.user_attrs.get('trailing_stop_count', '?')} ({best1.user_attrs.get('ts_pct_of_tickers', '?')}%)")

    best1_pt = json.loads(best1.user_attrs.get("per_ticker", "{}"))
    print_per_ticker(best1_pt, best_activation, best_ts_coarse)

    # ================================================================
    # ЭТАП 2
    # ================================================================
    print("\n" + "=" * 70)
    print(f"  ЭТАП 2: Доужение TS около {best_ts_coarse}%")
    print(f"          activation={best_activation}% (fixed)")
    ts_low = max(0.1, best_ts_coarse - 2.0)
    ts_high = best_ts_coarse + 2.0
    print(f"          TS: {ts_low:.1f}-{ts_high:.1f}%")
    print("=" * 70)

    study2 = optuna.create_study(
        study_name=f"{args.study_name}_phase2",
        direction="maximize",
        sampler=TPESampler(seed=43, multivariate=True),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        storage=storage_url,
        load_if_exists=True,
    )

    start2 = time.time()
    study2.optimize(
        lambda trial: objective_phase2(
            trial, tickers, days, timeout_sec, best_activation, best_ts_coarse, max_dd_limit),
        n_trials=args.n_trials_2,
        show_progress_bar=True,
    )
    elapsed2 = time.time() - start2

    best2 = study2.best_trial
    best_ts = best2.params["trailing_stop_pct"]

    print(f"\n  РЕЗУЛЬТАТ ЭТАПА 2 ({elapsed2:.0f}s):")
    print(f"  ✓ trailing_stop_pct = {best_ts}%")
    print(f"    Avg Profit:    {best2.user_attrs.get('avg_profit_pct', '?')}%")
    print(f"    Avg Max DD:    {best2.user_attrs.get('avg_max_dd_pct', '?')}%")
    print(f"    Profit Factor: {best2.user_attrs.get('profit_factor', '?')}")
    print(f"    Profitable:    {best2.user_attrs.get('profitable', '?')}")
    print(f"    TS triggers:   {best2.user_attrs.get('trailing_stop_count', '?')} ({best2.user_attrs.get('ts_pct_of_tickers', '?')}%)")

    best2_pt = json.loads(best2.user_attrs.get("per_ticker", "{}"))
    print_per_ticker(best2_pt, best_activation, best_ts)

    # ================================================================
    # ИТОГ
    # ================================================================
    print("\n" + "=" * 70)
    print("  ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 70)
    print(f"  equity_trailing_stop_activation_pct = {best_activation}%  (take-profit)")
    print(f"  equity_trailing_stop_pct            = {best_ts}%  (stop)")
    print(f"  max_drawdown_limit                  = {max_dd_limit}%  (fixed)")
    print(f"  equity_trailing_stop_timeout_sec     = {timeout_sec}s  (fixed)")
    print()
    print(f"  При ATH=87.75 (EPICUSDT real):")
    ath = 87.75
    init_cap = 80.0
    activation_tpv = init_cap * (1 + best_activation / 100)
    stop_tpv = ath * (1 - best_ts / 100)
    print(f"    Активация при TPV  = ${activation_tpv:.2f}")
    print(f"    Стоп при TPV       = ${stop_tpv:.2f} (от ATH)")
    print(f"    Прибыль на стопе   = ${stop_tpv - init_cap:+.2f}")
    if activation_tpv > 0:
        drain = activation_tpv - stop_tpv
        print(f"    Слив от активации  = ${drain:.2f} (было $10.61 при TS=15%)")

    output = {
        "optimal_activation_pct": best_activation,
        "optimal_trailing_stop_pct": best_ts,
        "max_drawdown_limit": max_dd_limit,
        "timeout_sec": timeout_sec,
        "phase1_results": {
            "avg_profit_pct": best1.user_attrs.get("avg_profit_pct"),
            "avg_max_dd_pct": best1.user_attrs.get("avg_max_dd_pct"),
            "profit_factor": best1.user_attrs.get("profit_factor"),
            "profitable": best1.user_attrs.get("profitable"),
            "per_ticker": best1_pt,
        },
        "phase2_results": {
            "avg_profit_pct": best2.user_attrs.get("avg_profit_pct"),
            "avg_max_dd_pct": best2.user_attrs.get("avg_max_dd_pct"),
            "profit_factor": best2.user_attrs.get("profit_factor"),
            "profitable": best2.user_attrs.get("profitable"),
            "per_ticker": best2_pt,
        },
        "config": {
            "tickers_count": len(tickers),
            "days": days,
            "trials_phase1": args.n_trials_1,
            "trials_phase2": args.n_trials_2,
        },
        "elapsed_sec": {"phase1": round(elapsed1, 1), "phase2": round(elapsed2, 1)},
    }
    output_path = PROJECT_DIR / "best_takeprofit_stop.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Сохранено: {output_path}")
    print(f"  Время: этап1={elapsed1:.0f}s, этап2={elapsed2:.0f}s")


if __name__ == "__main__":
    main()
