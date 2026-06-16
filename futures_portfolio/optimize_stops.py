"""
Optuna-оптимизация параметров стоплосса для SyntheticMarketNeutral портфеля.

Оптимизируемые параметры:
  - max_drawdown_limit                  (% от tpv_ath — high-water mark)
  - equity_trailing_stop_pct            (% просадки от ATH)
  - equity_trailing_stop_activation_pct (% роста от initial для активации)

НЕ оптимизируется (берётся из конфига):
  - equity_trailing_stop_timeout_sec

Стратегия: каждый trial тестирует ОДНУ комбинацию параметров на ВСЕХ
тикерах из config.json, метрика — средний net_profit_pct по тикерам.

Запуск:
  python optimize_stops.py
  python optimize_stops.py --n-trials 200
  python optimize_stops.py --storage optuna_stops.db --study-name stops_v1
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

# --- Пути проекта ---
PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR = PROJECT_DIR / "data"
STORAGE_PATH = PROJECT_DIR / "optuna_stops.db"

# --- Импорт бэктеста ---
sys.path.insert(0, str(PROJECT_DIR))
from backtest_rebalance import run_backtest

# --- Логирование ---
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("Optuna")
warnings.filterwarnings("ignore")

# ============================================================
# ПАРАМЕТРЫ ОПТИМИЗАЦИИ (timeout_sec убран — фиксированный)
# ============================================================
SEARCH_SPACE = {
    "max_drawdown_limit":                  (5.0, 50.0),
    "equity_trailing_stop_pct":            (2.0, 25.0),
    "equity_trailing_stop_activation_pct": (0.5, 10.0),
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tickers(cfg):
    """Читаем тикеры из tickers.txt (приоритет) или config.json."""
    tickers_file = PROJECT_DIR / "tickers.txt"
    if tickers_file.exists():
        with open(tickers_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return cfg.get("tickers", ["ALGOUSDT"])


def get_timeout_sec(cfg):
    return int(cfg.get("equity_trailing_stop_timeout_sec", 60))


def get_days(cfg):
    return float(cfg.get("backtest_period_days", 2.0))


def run_backtest_for_ticker(config, ticker, days):
    """Синхронный запуск бэктеста для одного тикера через patched config."""
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


def objective(trial: optuna.Trial) -> float:
    """
    Один trial = одна комбинация параметров, тестируемая на ВСЕХ тикерах.
    Возвращает средний net_profit_pct по всем тикерам.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cfg = load_config()
    tickers = get_tickers(cfg)
    days = get_days(cfg)
    timeout_sec = get_timeout_sec(cfg)

    # Предложение параметров (timeout_sec НЕ оптимизируем)
    params = {}
    for name, (low, high) in SEARCH_SPACE.items():
        if "max_drawdown" in name:
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, low, high, step=0.5)

    # Патч конфига
    cfg["max_drawdown_limit"] = params["max_drawdown_limit"]
    cfg["equity_trailing_stop_pct"] = params["equity_trailing_stop_pct"]
    cfg["equity_trailing_stop_activation_pct"] = params["equity_trailing_stop_activation_pct"]
    cfg["equity_trailing_stop_timeout_sec"] = timeout_sec  # фиксированный

    trial.set_user_attr("timeout_sec_fixed", timeout_sec)

    # Запуск бэктестов по тикерам (последовательно — asyncio внутри)
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

        profit = result.get("profit_pct", 0.0)
        dd = result.get("max_dd_pct", 0.0)
        ts = result.get("trailing_stop_triggered", False)
        cycles = result.get("cycles", 0)
        per_ticker[ticker] = {
            "profit": profit,
            "max_dd": dd,
            "trailing_stop": ts,
            "cycles": cycles,
        }

    if not per_ticker:
        raise optuna.TrialPruned()

    # Агрегация метрик
    n = len(per_tickers := per_ticker)
    avg_profit = sum(v["profit"] for v in per_ticker.values()) / n
    avg_dd = sum(v["max_dd"] for v in per_ticker.values()) / n
    max_dd = max(v["max_dd"] for v in per_ticker.values())
    ts_count = sum(1 for v in per_ticker.values() if v["trailing_stop"])
    total_cycles = sum(v["cycles"] for v in per_ticker.values())
    profitable = sum(1 for v in per_ticker.values() if v["profit"] > 0)

    # profit_factor аппроксимация
    pf = (avg_profit / avg_dd) if avg_dd > 0.1 else avg_profit / 0.1

    # User attrs — агрегаты + per-ticker детали
    trial.set_user_attr("avg_profit_pct", round(avg_profit, 4))
    trial.set_user_attr("avg_max_dd_pct", round(avg_dd, 2))
    trial.set_user_attr("max_dd_pct", round(max_dd, 2))
    trial.set_user_attr("trailing_stop_count", ts_count)
    trial.set_user_attr("trailing_stop_pct", round(ts_count / n * 100, 1))
    trial.set_user_attr("total_cycles", total_cycles)
    trial.set_user_attr("profitable_tickers", f"{profitable}/{n}")
    trial.set_user_attr("profit_factor", round(pf, 3))
    trial.set_user_attr("errors", errors)

    # Сохраняем per-ticker данные в JSON
    trial.set_user_attr("per_ticker", json.dumps(per_ticker))

    # --- Pruning ---
    trial.report(avg_profit, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # --- Штрафы ---

    # Если max drawdown любого тикера превысил лимит — штраф
    if max_dd > params["max_drawdown_limit"]:
        avg_profit *= 0.1

    # Слишком частое срабатывание trailing stop (>30% тикеров) — штраф
    if ts_count / n > 0.3:
        avg_profit *= 0.85

    # Меньше половины тикеров прибыльных — жёсткий штраф
    if profitable < n / 2:
        avg_profit *= 0.5

    # profit_factor < 1.0 — штраф
    if pf < 1.0:
        avg_profit *= 0.5

    return round(avg_profit, 4)


def main():
    parser = argparse.ArgumentParser(description="Оптимизация стоплоссов (Optuna, multi-ticker)")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None, help="Общий лимит времени (сек)")
    parser.add_argument("--storage", type=str, default=str(STORAGE_PATH),
                        help=f"SQLite DB (default: {STORAGE_PATH})")
    parser.add_argument("--study-name", type=str, default="stop_loss_multi")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("Backtest").setLevel(logging.INFO)

    # --- Инфо ---
    cfg = load_config()
    tickers = get_tickers(cfg)
    days = get_days(cfg)
    timeout_sec = get_timeout_sec(cfg)

    print("=" * 60)
    print("  Optuna Stop-Loss Optimization (multi-ticker)")
    print(f"  Tickers:  {len(tickers)} ({', '.join(tickers[:5])}...)")
    print(f"  Days:     {days} ({days * 24:.1f}h)")
    print(f"  Timeout:  {timeout_sec}s (fixed, not optimized)")
    print(f"  Trials:   {args.n_trials}")
    print(f"  Storage:  {args.storage}")
    print("=" * 60)

    # --- Study ---
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
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    elapsed = time.time() - start_time

    # --- РЕЗУЛЬТАТЫ ---
    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)

    best = study.best_trial
    best_pt = json.loads(best.user_attrs.get("per_ticker", "{}"))

    print(f"\n  Лучший trial #{best.number}")
    print(f"  Avg Profit:       {best.user_attrs.get('avg_profit_pct', '?')}%")
    print(f"  Avg Max DD:       {best.user_attrs.get('avg_max_dd_pct', '?')}%")
    print(f"  Worst Max DD:     {best.user_attrs.get('max_dd_pct', '?')}%")
    print(f"  Profit Factor:    {best.user_attrs.get('profit_factor', '?')}")
    print(f"  Profitable:       {best.user_attrs.get('profitable_tickers', '?')}")
    print(f"  Trailing Stops:   {best.user_attrs.get('trailing_stop_count', '?')} "
          f"({best.user_attrs.get('trailing_stop_pct', '?')}%)")
    print(f"  Total Cycles:     {best.user_attrs.get('total_cycles', '?')}")
    print(f"  Timeout (fixed):  {best.user_attrs.get('timeout_sec_fixed', '?')}s")
    print(f"  Errors:           {best.user_attrs.get('errors', 0)}")

    print(f"\n  Параметры стоплосса:")
    for name in SEARCH_SPACE:
        val = best.params[name]
        print(f"    {name:<40s} = {val}")

    # Per-ticker детали лучшего trial
    if best_pt:
        print(f"\n  Per-ticker (best trial):")
        for ticker in sorted(best_pt.keys()):
            v = best_pt[ticker]
            ts = "TS" if v["trailing_stop"] else "  "
            print(f"    {ticker:<14s} profit={v['profit']:>+8.3f}%  dd={v['max_dd']:>6.2f}%  "
                  f"cyc={v['cycles']:>3d}  {ts}")

    # Top-5
    print(f"\n  Top-5 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:5]
    for rank, t in enumerate(top5, 1):
        prof = t.user_attrs.get("profitable_tickers", "?")
        ts_pct = t.user_attrs.get("trailing_stop_pct", "?")
        dd = t.user_attrs.get("avg_max_dd_pct", "?")
        print(f"    #{rank} Trial {t.number:>3d}: avg_profit={t.value:>8.3f}%  "
              f"avg_dd={dd:>5}%  profitable={prof}  TS={ts_pct}%")

    # --- Сохранение ---
    best_params = {name: best.params[name] for name in SEARCH_SPACE}

    output_path = PROJECT_DIR / "best_stops.json"
    output = {
        "config": {
            "tickers_count": len(tickers),
            "days": days,
            "timeout_sec_fixed": timeout_sec,
            "equity_trailing_stop_timeout_sec": timeout_sec,
        },
        "trials_total": len(study.trials),
        "trials_completed": len(completed),
        "elapsed_sec": round(elapsed, 1),
        "avg_profit_pct": best.user_attrs.get("avg_profit_pct"),
        "avg_max_dd_pct": best.user_attrs.get("avg_max_dd_pct"),
        "worst_max_dd_pct": best.user_attrs.get("max_dd_pct"),
        "profit_factor": best.user_attrs.get("profit_factor"),
        "profitable_tickers": best.user_attrs.get("profitable_tickers"),
        "trailing_stop_pct": best.user_attrs.get("trailing_stop_pct"),
        "params": best_params,
        "per_ticker": best_pt,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Сохранено: {output_path}")
    print(f"  Время: {elapsed:.1f}s")

    return study


if __name__ == "__main__":
    main()
