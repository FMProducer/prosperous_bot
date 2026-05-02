import asyncio
import json
import os
import logging
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

# Import functions from existing script
from backtest_rebalance import run_backtest

# Define threshold range for optimization
THRESHOLDS = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("Optimizer")

def run_backtest_sync(config_path: str, data_dir: str, ticker: str, threshold: float):
    """Sync wrapper to run backtest in a separate process."""
    # We use asyncio.run because run_backtest is async
    try:
        # We need to suppress logs inside processes to keep output clean
        return asyncio.run(run_backtest(
            config_path=config_path,
            data_dir=data_dir,
            live_mode=False,
            ticker_override=ticker,
            threshold_override=threshold,
            quiet=True
        ))
    except Exception as e:
        return None

async def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    tickers = config.get("tickers", [])
    if not tickers:
        logger.error("No tickers found in config.")
        return

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    logger.info(f"Starting threshold optimization for {len(tickers)} tickers...")
    logger.info(f"Thresholds to test: {THRESHOLDS}")

    best_threshold = config["portfolios"][0].get("rebalance_threshold", 0.02)
    max_avg_profit = -float('inf')
    
    # We will run backtests in parallel using ProcessPoolExecutor
    # Optimization: One threshold at a time, parallel across tickers
    # OR: All combinations in parallel.
    
    results_map = {} # threshold -> [profits]

    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        loop = asyncio.get_running_loop()
        
        for threshold in THRESHOLDS:
            logger.info(f"Testing threshold: {threshold:.4f}...")
            
            # Parallelize across tickers for this threshold
            tasks = [
                loop.run_in_executor(executor, run_backtest_sync, config_path, data_dir, ticker, threshold)
                for ticker in tickers
            ]
            
            ticker_results = await asyncio.gather(*tasks)
            
            profits = [res["profit_pct"] for res in ticker_results if res]
            if profits:
                avg_profit = sum(profits) / len(profits)
                results_map[threshold] = avg_profit
                logger.info(f"  Result -> Avg Profit: {avg_profit:+.4f}% | Max DD: {max([res['max_dd_pct'] for res in ticker_results if res]):.2f}%")
            else:
                logger.warning(f"  No valid results for threshold {threshold}")

    if not results_map:
        logger.error("Optimization failed: no results collected.")
        return

    best_threshold = max(results_map, key=results_map.get)
    best_profit = results_map[best_threshold]
    
    logger.info("=" * 50)
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"Best Universal Threshold: {best_threshold:.4f}")
    logger.info(f"Expected Avg Profit (3h): {best_profit:+.4f}%")
    logger.info("=" * 50)

    # Update configuration
    config["portfolios"][0]["rebalance_threshold"] = best_threshold
    
    # Update ticker-specific thresholds if they exist to keep things uniform
    if "ticker_thresholds" in config["portfolios"][0]:
        for ticker in tickers:
            config["portfolios"][0]["ticker_thresholds"][ticker] = best_threshold

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Updated {config_path} with new threshold.")

if __name__ == "__main__":
    asyncio.run(main())
