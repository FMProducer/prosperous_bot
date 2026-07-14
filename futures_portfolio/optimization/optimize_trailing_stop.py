"""
Optuna-оптимизация параметров трейлинг стопа.

Оптимизируемые параметры (все 3):
  - equity_trailing_stop_pct            (0.1 - 6.0) — шаг 0.1
  - equity_trailing_stop_activation_pct (3.0 - 10.0) — шаг 0.5
  - equity_trailing_stop_timeout_sec    (10 - 120) — шаг 10

НЕ оптимизируется (фиксировано):
  - max_drawdown_limit: 100.0

Стратегия: каждый trial = одна комбинация на ВСЕХ тикерах из tickers.txt.
Метрика: avg_profit_pct с штрафами за высокий DD, частые TS, низкий profit_factor.

Запуск:
  python optimize_trailing_stop.py
  python optimize_trailing_stop.py --n-trials 300
  python optimize_trailing_stop.py --study-name ts_3param
"""

import asyncio
import json
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_DIR = PROJECT_DIR / "data"
TICKERS_FILE = PROJECT_DIR / "tickers.txt"
STORAGE_PATH = PROJECT_DIR / "optuna_trailing_stop.db"

sys.path.insert(0, str(PROJECT_DIR.parent))
from backtest.backtest_rebalance import run_backtest

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("Optuna-TS")
warnings.filterwarnings("ignore")

# Глобальная переменная для days (устанавливается в main())
args_days = 3.0

# ============================================================
# SEARCH SPACE — только 3 параметра trailing stop
# ============================================================
SEARCH_SPACE = {
    "equity_trailing_stop_pct":            (0.3, 6.0, 0.1),    # low, high, step
}

# Фиксированные параметры
FIXED_ACTIVATION_PCT = 0.0
FIXED_TIMEOUT_SEC = 60


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tickers():
    if TICKERS_FILE.exists():
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    cfg = load_config()
    return cfg.get("tickers", ["ALGOUSDT"])


def get_days(cfg):
    return float(cfg.get("backtest_period_days", 2.0))


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


def objective(trial: optuna.Trial) -> float:
    cfg = load_config()
    tickers = get_tickers()
    days = args_days

    # Фиксируем max_drawdown_limit
    cfg["max_drawdown_limit"] = 100.0

    # Оптимизируем ТОЛЬКО equity_trailing_stop_pct
    ts_pct = trial.suggest_float(
        "equity_trailing_stop_pct",
        SEARCH_SPACE["equity_trailing_stop_pct"][0],
        SEARCH_SPACE["equity_trailing_stop_pct"][1],
        step=SEARCH_SPACE["equity_trailing_stop_pct"][2],
    )

    # Фиксированные параметры
    ts_act = FIXED_ACTIVATION_PCT
    ts_timeout = FIXED_TIMEOUT_SEC

    cfg["equity_trailing_stop_pct"] = ts_pct
    cfg["equity_trailing_stop_activation_pct"] = ts_act
    cfg["equity_trailing_stop_timeout_sec"] = ts_timeout

    trial.set_user_attr("ts_pct", ts_pct)
    trial.set_user_attr("ts_activation_fixed", ts_act)
    trial.set_user_attr("ts_timeout_fixed", ts_timeout)

    # Запуск бэктестов
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
        tg_blocks = result.get("trend_guard_blocks", 0)
        vg_blocks = result.get("velocity_guard_blocks", 0)
        sortino = result.get("sortino_ratio", 0.0)
        per_ticker[ticker] = {
            "profit": profit,
            "max_dd": dd,
            "trailing_stop": ts,
            "cycles": cycles,
            "tg_blocks": tg_blocks,
            "vg_blocks": vg_blocks,
            "sortino": sortino,
        }

    if not per_ticker:
        raise optuna.TrialPruned()

    n = len(per_ticker)
    avg_profit = sum(v["profit"] for v in per_ticker.values()) / n
    avg_dd = sum(v["max_dd"] for v in per_ticker.values()) / n
    max_dd = max(v["max_dd"] for v in per_ticker.values())
    avg_sortino = sum(v.get("sortino", 0.0) for v in per_ticker.values()) / n
    ts_count = sum(1 for v in per_ticker.values() if v["trailing_stop"])
    total_cycles = sum(v["cycles"] for v in per_ticker.values())
    total_tg_blocks = sum(v.get("tg_blocks", 0) for v in per_ticker.values())
    total_vg_blocks = sum(v.get("vg_blocks", 0) for v in per_ticker.values())
    total_guard_blocks = total_tg_blocks + total_vg_blocks
    profitable = sum(1 for v in per_ticker.values() if v["profit"] > 0)

    pf = (avg_profit / avg_dd) if avg_dd > 0.1 else avg_profit / 0.1

    trial.set_user_attr("avg_profit_pct", round(avg_profit, 4))
    trial.set_user_attr("avg_max_dd_pct", round(avg_dd, 2))
    trial.set_user_attr("max_dd_pct", round(max_dd, 2))
    trial.set_user_attr("avg_sortino", round(avg_sortino, 4))
    trial.set_user_attr("trailing_stop_count", ts_count)
    trial.set_user_attr("trailing_stop_pct", round(ts_count / n * 100, 1))
    trial.set_user_attr("total_cycles", total_cycles)
    trial.set_user_attr("total_tg_blocks", total_tg_blocks)
    trial.set_user_attr("total_vg_blocks", total_vg_blocks)
    trial.set_user_attr("total_guard_blocks", total_guard_blocks)
    trial.set_user_attr("profitable_tickers", f"{profitable}/{n}")
    trial.set_user_attr("profit_factor", round(pf, 3))
    trial.set_user_attr("errors", errors)
    trial.set_user_attr("per_ticker", json.dumps(per_ticker))

    trial.report(avg_profit, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # --- Целевая метрика: Sortino Ratio ---
    # Штрафует за глубокие просадки, но не штрафует за рост
    # Чем выше — тем лучше
    score = avg_sortino

    # Менее 70% прибыльных — жёсткий штраф
    if profitable < n * 0.7:
        score *= 0.3

    # profit_factor < 1.0 — штраф
    if pf < 1.0:
        score *= 0.3

    # Минимальная активность: < 10 циклов на тикер — подозрительно
    avg_cycles = total_cycles / n if n > 0 else 0
    if avg_cycles < 10:
        score *= 0.3

    return round(score, 4)


def main():
    parser = argparse.ArgumentParser(description="Optuna: оптимизация trailing stop (1 параметр)")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=None, help="Общий лимит времени (сек)")
    parser.add_argument("--storage", type=str, default=str(STORAGE_PATH))
    parser.add_argument("--study-name", type=str, default="trailing_stop_1param")
    parser.add_argument("--days", type=float, default=3.0, help="Дни бэктеста (default: 3.0)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    tickers = get_tickers()
    days = args.days

    # Делаем days доступным для objective
    global args_days
    args_days = days

    print("=" * 60)
    print("  Optuna Trailing Stop Optimization (1 parameter: ts_pct)")
    print(f"  Tickers:  {len(tickers)} ({', '.join(tickers[:5])}...)")
    print(f"  Days:     {days} ({days * 24:.1f}h)")
    print(f"  Trials:   {args.n_trials}")
    print(f"  Storage:  {args.storage}")
    print(f"  Fixed:    activation={FIXED_ACTIVATION_PCT}%, timeout={FIXED_TIMEOUT_SEC}s")
    print(f"  Search space:")
    for name, (low, high, step) in SEARCH_SPACE.items():
        print(f"    {name:<40s} [{low} .. {high}] step={step}")
    print("=" * 60)

    storage_url = f"sqlite:///{args.storage}"
    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=0)

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
    print(f"  Avg Sortino:      {best.user_attrs.get('avg_sortino', '?')}")
    print(f"  Avg Max DD:       {best.user_attrs.get('avg_max_dd_pct', '?')}%")
    print(f"  Worst Max DD:     {best.user_attrs.get('max_dd_pct', '?')}%")
    print(f"  Profit Factor:    {best.user_attrs.get('profit_factor', '?')}")
    print(f"  Profitable:       {best.user_attrs.get('profitable_tickers', '?')}")
    print(f"  Trailing Stops:   {best.user_attrs.get('trailing_stop_count', '?')} "
          f"({best.user_attrs.get('trailing_stop_pct', '?')}%)")
    print(f"  Total Cycles:     {best.user_attrs.get('total_cycles', '?')}")
    print(f"  TG+VG Blocks:     {best.user_attrs.get('total_guard_blocks', '?')} "
          f"(TG={best.user_attrs.get('total_tg_blocks', '?')}, VG={best.user_attrs.get('total_vg_blocks', '?')})")
    print(f"  Errors:           {best.user_attrs.get('errors', 0)}")

    print(f"\n  Параметры trailing stop:")
    print(f"    equity_trailing_stop_pct            = {best.params['equity_trailing_stop_pct']}")
    print(f"    equity_trailing_stop_activation_pct = {FIXED_ACTIVATION_PCT} (fixed)")
    print(f"    equity_trailing_stop_timeout_sec    = {FIXED_TIMEOUT_SEC} (fixed)")

    if best_pt:
        print(f"\n  Per-ticker (best trial):")
        for ticker in sorted(best_pt.keys()):
            v = best_pt[ticker]
            ts = "TS" if v["trailing_stop"] else "  "
            print(f"    {ticker:<14s} profit={v['profit']:>+8.3f}%  dd={v['max_dd']:>6.2f}%  "
                  f"cyc={v['cycles']:>3d}  {ts}")

    # Top-10
    print(f"\n  Top-10 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top10 = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:10]
    for rank, t in enumerate(top10, 1):
        prof = t.user_attrs.get("profitable_tickers", "?")
        ts_pct_val = t.user_attrs.get("trailing_stop_pct", "?")
        dd = t.user_attrs.get("avg_max_dd_pct", "?")
        cyc = t.user_attrs.get("total_cycles", "?")
        print(f"    #{rank:>2d} Trial {t.number:>4d}: score={t.value:>8.3f}  "
              f"profit={t.user_attrs.get('avg_profit_pct', '?'):>8}%  "
              f"sortino={t.user_attrs.get('avg_sortino', '?'):>7}  "
              f"dd={dd:>5}%  profitable={prof}  TS={ts_pct_val}%  "
              f"cycles={cyc}  ts_pct={t.params['equity_trailing_stop_pct']:.1f}")

    # --- Сохранение ---
    output_path = PROJECT_DIR / "best_trailing_stop.json"
    output = {
        "config": {
            "tickers_count": len(tickers),
            "days": days,
            "max_drawdown_limit": 100.0,
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
        "params": {
            "equity_trailing_stop_pct": best.params["equity_trailing_stop_pct"],
            "equity_trailing_stop_activation_pct": FIXED_ACTIVATION_PCT,
            "equity_trailing_stop_timeout_sec": FIXED_TIMEOUT_SEC,
        },
        "per_ticker": best_pt,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Сохранено: {output_path}")
    print(f"  Время: {elapsed:.1f}s")

    return study


if __name__ == "__main__":
    main()
