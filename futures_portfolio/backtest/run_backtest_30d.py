"""
run_backtest_30d.py — Бэктест на 30 днях для всех тикеров.
Trailing stop отключен в config.json.
"""
import asyncio
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.backtest_rebalance import run_backtest

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TICKERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tickers.txt")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

DAYS = 30.0
MAX_DD_PCT = 15.0
MIN_CYCLES = 20
MIN_PROFIT = 0.0


async def run_one(ticker: str) -> dict | None:
    src_30d = os.path.join(DATA_DIR, f"{ticker}_live_30d.feather")
    src_live = os.path.join(DATA_DIR, f"{ticker}_live.feather")

    if not os.path.exists(src_30d):
        return None

    try:
        # Копируем 30d -> live (run_backtest без live_mode читает live)
        df = pd.read_feather(src_30d)
        df.to_feather(src_live)

        res = await run_backtest(
            config_path=CONFIG_PATH,
            data_dir=DATA_DIR,
            ticker_override=ticker,
            live_mode=False,
            days=DAYS,
            quiet=True
        )
        return res
    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None


async def main():
    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    available = [t for t in tickers if os.path.exists(os.path.join(DATA_DIR, f"{t}_live_30d.feather"))]

    print("=" * 70)
    print(f"  30-Day Backtest | {len(available)}/{len(tickers)} tickers")
    print(f"  Filters: MaxDD < {MAX_DD_PCT}%, Cycles >= {MIN_CYCLES}, Profit > {MIN_PROFIT}%")
    print("=" * 70)
    print()

    results = []
    failed = []

    for i, ticker in enumerate(available, 1):
        print(f"[{i:>2d}/{len(available)}] {ticker}...", end=" ", flush=True)
        res = await run_one(ticker)
        if res:
            results.append({"ticker": ticker, **res})
            flag = "PASS" if (res["max_dd_pct"] < MAX_DD_PCT and res["cycles"] >= MIN_CYCLES and res["profit_pct"] > MIN_PROFIT) else "FAIL"
            print(f"Profit={res['profit_pct']:+.2f}% DD={res['max_dd_pct']:.2f}% Cyc={res['cycles']} Liq={res.get('liquidations',0)} [{flag}]")
        else:
            failed.append(ticker)
            print("SKIPPED (no 30d data)")

    # Results
    results.sort(key=lambda x: x["profit_pct"], reverse=True)

    print()
    print("=" * 70)
    print("  RESULTS (sorted by profit)")
    print("=" * 70)
    print(f"{'Ticker':<18s} {'Profit':>9s} {'MaxDD':>7s} {'Cycles':>7s} {'Liq':>4s} {'TG':>5s}")
    print("-" * 60)

    passed = []
    for r in results:
        is_pass = (r["max_dd_pct"] < MAX_DD_PCT and r["cycles"] >= MIN_CYCLES and r["profit_pct"] > MIN_PROFIT)
        marker = " [PASS]" if is_pass else ""
        if is_pass:
            passed.append(r)
        print(f"{r['ticker']:<18s} {r['profit_pct']:>+8.2f}% {r['max_dd_pct']:>6.2f}% {r['cycles']:>6d} {r.get('liquidations',0):>3d} {r.get('trend_guard_blocks',0):>4d}{marker}")

    print()
    print("REJECTED:")
    for r in results:
        if r not in passed:
            reasons = []
            if r["max_dd_pct"] >= MAX_DD_PCT:
                reasons.append(f"DD={r['max_dd_pct']:.1f}%")
            if r["cycles"] < MIN_CYCLES:
                reasons.append(f"Cyc={r['cycles']}")
            if r["profit_pct"] <= MIN_PROFIT:
                reasons.append(f"Profit={r['profit_pct']:+.2f}%")
            print(f"  {r['ticker']:<18s} {', '.join(reasons)}")

    n = len(results)
    n_pass = len(passed)
    if passed:
        avg_p = sum(r["profit_pct"] for r in passed) / n_pass
        avg_dd = sum(r["max_dd_pct"] for r in passed) / n_pass
        total_liq = sum(r.get("liquidations", 0) for r in passed)
    else:
        avg_p = avg_dd = total_liq = 0

    print()
    print("=" * 70)
    print(f"  SUMMARY")
    print(f"=" * 70)
    print(f"  Tested:     {n}")
    print(f"  Passed:     {n_pass}")
    print(f"  Rejected:   {n - n_pass}")
    if failed:
        print(f"  No data:    {len(failed)}")
    if passed:
        print(f"  Avg Profit: {avg_p:+.2f}%")
        print(f"  Avg DD:     {avg_dd:.2f}%")
        print(f"  Total Liq:  {total_liq}")
    print("=" * 70)

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_30d_results.json")
    with open(output_path, "w") as f:
        json.dump({"passed": passed, "rejected": [r for r in results if r not in passed]}, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
