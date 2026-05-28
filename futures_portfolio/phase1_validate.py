#!/usr/bin/env python3
"""
Phase 1 Batch Validation Script
Runs backtest_rebalance.py for top-10 tickers on multiple time periods
and produces a comparison CSV with verdicts.
"""

import subprocess
import re
import csv
import sys
import time
import requests

# Configuration
PYTHON_EXE = r"C:\Python\Prosperous_Bot\.venv\Scripts\python.exe"
WORK_DIR = r"C:\Python\Prosperous_Bot\futures_portfolio"
OUTPUT_CSV = r"C:\Python\Prosperous_Bot\futures_portfolio\phase1_results.csv"

TICKERS = [
    "ZEREBROUSDT", "KOMAUSDT", "DODOXUSDT", "JELLYJELLYUSDT",
    "GMTUSDT", "HMSTRUSDT", "SEIUSDT", "GRASSUSDT", "RSRUSDT", "INJUSDT"
]
DAYS_LIST = [2, 7]

# Binance Futures API
BINANCE_24HR_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"

TIMEOUT_SECONDS = 300  # 5 min per backtest


def fetch_volume_24h(ticker):
    """Fetch 24h quoteVolume from Binance Futures API."""
    try:
        resp = requests.get(BINANCE_24HR_URL, params={"symbol": ticker}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("quoteVolume", 0))
    except Exception as e:
        print(f"  [WARN] Volume fetch failed for {ticker}: {e}")
        return 0.0


def parse_backtest_output(output):
    """Parse backtest output lines and extract metrics."""
    metrics = {
        "profit_pct": None,
        "max_dd_pct": None,
        "cycles": None,
        "asset_chg_pct": None,
        "liquidations": None,
    }

    for line in output.splitlines():
        # Line: "Profit: +1.23% | MaxDD: 4.56% | Cycles: 12"
        m = re.search(
            r"Profit:\s*([+\-]?\d+\.?\d*)%\s*\|\s*"
            r"MaxDD:\s*(\d+\.?\d*)%\s*\|\s*"
            r"Cycles:\s*(\d+)",
            line,
        )
        if m:
            metrics["profit_pct"] = float(m.group(1))
            metrics["max_dd_pct"] = float(m.group(2))
            metrics["cycles"] = int(m.group(3))

        # Line: "Asset Change: +7.89% | SAFE Reserve: 10.50 USDT"
        m = re.search(r"Asset Change:\s*([+\-]?\d+\.?\d*)%", line)
        if m:
            metrics["asset_chg_pct"] = float(m.group(1))

        # Line: "Skipped Expansions: 3 | Liquidations: 0"
        m = re.search(r"Liquidations:\s*(\d+)", line)
        if m:
            metrics["liquidations"] = int(m.group(1))

    # Check that all metrics were found
    missing = [k for k, v in metrics.items() if v is None]
    if missing:
        print(f"  [WARN] Could not parse: {missing}")
        print(f"  --- Raw output (last 500 chars) ---")
        print(output[-500:])

    return metrics


def run_backtest(ticker, days):
    """Run backtest_rebalance.py via subprocess for a given ticker and days."""
    cmd = [
        PYTHON_EXE,
        "backtest_rebalance.py",
        "--ticker", ticker,
        "--days", str(days),
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=WORK_DIR,
        )
        full_output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"  [ERROR] Backtest exited with code {result.returncode}")
            print(f"  --- stderr ---")
            print(result.stderr[-300:] if result.stderr else "(empty)")
        return full_output
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Backtest timed out after {TIMEOUT_SECONDS}s")
        return ""
    except Exception as e:
        print(f"  [ERROR] Backtest failed: {e}")
        return ""


def compute_verdict(profit_pct, max_dd_pct, liquidations, volume_24h):
    """Compute PASS/FAIL verdict based on criteria."""
    if (profit_pct is not None
            and profit_pct > 0.5
            and max_dd_pct is not None
            and max_dd_pct < 10
            and liquidations is not None
            and liquidations == 0
            and volume_24h > 5_000_000):
        return "PASS"
    return "FAIL"


def format_pct(val):
    """Format a percentage value."""
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


def format_float(val, decimals=2):
    """Format a float value."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def format_volume(val):
    """Format volume in millions."""
    if val is None:
        return "N/A"
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    elif val >= 1_000:
        return f"{val / 1_000:.1f}K"
    else:
        return f"{val:.0f}"


def main():
    print("=" * 80)
    print("  Phase 1 Batch Validation")
    print("  Running backtest_rebalance.py for top-10 tickers")
    print("=" * 80)

    # Fetch volumes first (one per ticker)
    print("\n--- Fetching 24h volumes from Binance Futures API ---")
    volumes = {}
    for ticker in TICKERS:
        vol = fetch_volume_24h(ticker)
        volumes[ticker] = vol
        print(f"  {ticker}: ${vol:,.0f}")
        time.sleep(0.25)  # Rate limit courtesy

    # Run backtests
    print("\n--- Running Backtests ---")
    results = []
    total = len(TICKERS) * len(DAYS_LIST)
    idx = 0

    for ticker in TICKERS:
        for days in DAYS_LIST:
            idx += 1
            print(f"\n[{idx}/{total}] {ticker} | {days}d")
            raw_output = run_backtest(ticker, days)
            metrics = parse_backtest_output(raw_output)

            volume = volumes.get(ticker, 0)
            verdict = compute_verdict(
                metrics.get("profit_pct"),
                metrics.get("max_dd_pct"),
                metrics.get("liquidations"),
                volume,
            )

            results.append({
                "ticker": ticker,
                "days": days,
                "profit_pct": metrics.get("profit_pct"),
                "max_dd_pct": metrics.get("max_dd_pct"),
                "cycles": metrics.get("cycles"),
                "asset_chg": metrics.get("asset_chg_pct"),
                "liquidations": metrics.get("liquidations"),
                "volume_24h_usdt": round(volume, 2),
                "verdict": verdict,
            })

    # Write CSV
    print(f"\n--- Writing CSV: {OUTPUT_CSV} ---")
    fieldnames = [
        "ticker", "days", "profit_pct", "max_dd_pct", "cycles",
        "asset_chg", "liquidations", "volume_24h_usdt", "verdict",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"  Wrote {len(results)} rows.")

    # Print Summary Table
    print("\n" + "=" * 110)
    print("  Phase 1 Validation Results Summary")
    print("=" * 110)

    # Header
    header = (
        f"{'#':<3} {'Ticker':<18} {'Days':<5} {'Profit':<10} {'MaxDD':<8} "
        f"{'Cycles':<7} {'AssetChg':<10} {'Liq':<4} {'Volume':<10} {'Verdict'}"
    )
    print(header)
    print("-" * 110)

    pass_count = 0
    for i, row in enumerate(results, 1):
        verdict_marker = ""
        if row["verdict"] == "PASS":
            pass_count += 1
            verdict_marker = "<-- PASS"

        line = (
            f"{i:<3} {row['ticker']:<18} {row['days']:<5} "
            f"{format_pct(row['profit_pct']):<10} "
            f"{format_float(row['max_dd_pct'], 2):<8} "
            f"{str(row['cycles'] if row['cycles'] is not None else 'N/A'):<7} "
            f"{format_pct(row['asset_chg']):<10} "
            f"{str(row['liquidations'] if row['liquidations'] is not None else 'N/A'):<4} "
            f"{format_volume(row['volume_24h_usdt'] if row['volume_24h_usdt'] else 0):<10} "
            f"{row['verdict']} {verdict_marker}"
        )
        print(line)

    print("-" * 110)
    print(f"  Total: {len(results)} tests | {pass_count} PASS | {len(results) - pass_count} FAIL")
    print()

    # Breakdown by ticker
    print("--- Per-Ticker Summary ---")
    for ticker in TICKERS:
        ticker_results = [r for r in results if r["ticker"] == ticker]
        ticker_pass = sum(1 for r in ticker_results if r["verdict"] == "PASS")
        best = max(
            [r["profit_pct"] for r in ticker_results if r["profit_pct"] is not None],
            default=None,
        )
        best_str = f"{best:+.2f}%" if best is not None else "N/A"
        print(f"  {ticker:<18} | Pass: {ticker_pass}/{len(ticker_results)} | Best Profit: {best_str}")

    print("\n  Criteria for PASS:")
    print("    profit_pct > 0.5%  AND  max_dd_pct < 10%  AND  liquidations == 0  AND  volume_24h > $5M")
    print(f"\n  CSV saved to: {OUTPUT_CSV}")
    print("=" * 110)


if __name__ == "__main__":
    main()
