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

    # Sort and print
    results.sort(key=lambda x: x['profit_pct'], reverse=True)
    print("\nResults (Sorted by Profit):")
    for r in results:
        print(f"{r['ticker']}: {r['profit_pct']:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
