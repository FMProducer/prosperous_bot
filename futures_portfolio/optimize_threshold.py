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

# Define threshold range for optimization - more realistic for rebalancing
THRESHOLDS = [0.002, 0.004, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("Optimizer")

def run_backtest_sync(config_path: str, data_dir: str, ticker: str, threshold: float):
    """Sync wrapper to run backtest in a separate process."""
    try:
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

    # Results: threshold -> {avg_profit, avg_cycles}
    results_map = {} 

    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        loop = asyncio.get_running_loop()
        
        for threshold in THRESHOLDS:
            logger.info(f"Testing threshold: {threshold:.4f}...")
            
            tasks = [
                loop.run_in_executor(executor, run_backtest_sync, config_path, data_dir, ticker, threshold)
                for ticker in tickers
            ]
            
            ticker_results = [res for res in await asyncio.gather(*tasks) if res]
            
            if ticker_results:
                avg_profit = sum(res["profit_pct"] for res in ticker_results) / len(ticker_results)
                avg_cycles = sum(res["cycles"] for res in ticker_results) / len(ticker_results)
                max_dd = max(res['max_dd_pct'] for res in ticker_results)
                
                results_map[threshold] = {
                    "profit": avg_profit,
                    "cycles": avg_cycles,
                    "max_dd": max_dd
                }
                logger.info(f"  Result -> Avg Profit: {avg_profit:+.4f}% | Avg Cycles: {avg_cycles:.1f} | Max DD: {max_dd:.2f}%")
            else:
                logger.warning(f"  No valid results for threshold {threshold}")

    if not results_map:
        logger.error("Optimization failed: no results collected.")
        return

    # Selection Logic: 
    # 1. We MUST have rebalances. If avg_cycles < 1.0, it's a "static" threshold, not rebalancing.
    # 2. Among those with cycles >= 1, pick the one with highest profit.
    # 3. If NONE have cycles >= 1, pick the one with highest cycles (attempting to rebalance).
    
    valid_candidates = {t: v for t, v in results_map.items() if v["cycles"] >= 1.0}
    
    if valid_candidates:
        best_threshold = max(valid_candidates, key=lambda k: valid_candidates[k]["profit"])
    else:
        # Fallback: find the one that rebalances even a little bit
        best_threshold = max(results_map, key=lambda k: results_map[k]["cycles"])
        logger.warning("No threshold achieved avg cycles >= 1.0. Picking threshold with maximum activity.")

    best_profit = results_map[best_threshold]["profit"]
    best_cycles = results_map[best_threshold]["cycles"]
    
    logger.info("=" * 50)
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"Best Universal Threshold: {best_threshold:.4f}")
    logger.info(f"Expected Avg Profit: {best_profit:+.4f}%")
    logger.info(f"Expected Avg Cycles: {best_cycles:.1f}")
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
