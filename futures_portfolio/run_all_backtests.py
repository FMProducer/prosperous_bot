import asyncio
import json
from backtest_rebalance import run_backtest
import os

async def main():
    if not os.path.exists("tickers.txt"):
        print("tickers.txt not found.")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    results = []
    # Assuming the script runs from 'futures_portfolio'
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    print(f"Starting backtests for {len(tickers)} tickers...")
    
    for ticker in tickers:
        print(f"Running backtest for {ticker}...")
        try:
            res = await run_backtest(
            config_path="config.json",
            data_dir=data_dir,
            ticker_override=ticker,
            live_mode=True,
            days=2.0,
            quiet=True
            )
            if res:
                res['ticker'] = ticker
                results.append(res)
                print(f"Finished {ticker}: Profit {res['profit_pct']:.2f}%")
            else:
                print(f"Failed {ticker}")
        except Exception as e:
            print(f"Error running backtest for {ticker}: {e}")

    # Sort and print top 100
    results.sort(key=lambda x: x['profit_pct'], reverse=True)
    
    print("\n" + "=" * 60)
    print("Top 100 Results (Sorted by Profit):")
    print("=" * 60)
    print(f"{'Ticker':<18s} {'Profit':>8s} {'MaxDD':>7s} {'Cycles':>7s} {'Liq':>4s} {'TG':>5s}")
    print("-" * 100)
    for r in results[:100]:
        liq = r.get('liquidations', 0)
        tg = r.get('trend_guard_blocks', 0)
        ts = " TS" if r.get('trailing_stop_triggered') else ""
        print(f"{r['ticker']:<18s} {r['profit_pct']:>+7.2f}% {r['max_dd_pct']:>6.2f}% {r['cycles']:>6d} {liq:>3d} {tg:>4d}{ts}")

    # Bottom 20
    print("\n" + "=" * 60)
    print("Bottom 20 Results (Worst Performers):")
    print("=" * 60)
    print(f"{'Ticker':<18s} {'Profit':>8s} {'MaxDD':>7s} {'Cycles':>7s} {'Liq':>4s} {'TG':>5s}")
    print("-" * 100)
    for r in results[-20:]:
        liq = r.get('liquidations', 0)
        tg = r.get('trend_guard_blocks', 0)
        ts = " TS" if r.get('trailing_stop_triggered') else ""
        print(f"{r['ticker']:<18s} {r['profit_pct']:>+7.2f}% {r['max_dd_pct']:>6.2f}% {r['cycles']:>6d} {liq:>3d} {tg:>4d}{ts}")

    # Summary
    n = len(results)
    profitable = sum(1 for r in results if r['profit_pct'] > 0)
    avg_profit = sum(r['profit_pct'] for r in results) / n if n else 0
    avg_dd = sum(r['max_dd_pct'] for r in results) / n if n else 0
    total_liqs = sum(r.get('liquidations', 0) for r in results)
    total_tg = sum(r.get('trend_guard_blocks', 0) for r in results)
    
    print("\n" + "=" * 60)
    print(f"Summary: {n} tickers | Profitable: {profitable}/{n} ({profitable/n*100:.0f}%)")
    print(f"Avg Profit: {avg_profit:+.2f}% | Avg MaxDD: {avg_dd:.2f}% | Total Liqs: {total_liqs} | Total TG blocks: {total_tg}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
