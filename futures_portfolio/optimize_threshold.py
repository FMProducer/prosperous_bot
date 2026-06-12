import asyncio
import json
import os
import logging
import sys
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

# Import functions from existing script
from backtest_rebalance import run_backtest

# Define asymmetric threshold pairs for optimization (surplus, deficit)
# Surplus: low threshold (aggressive profit taking)
# Deficit: higher threshold (patient averaging on dips)
THRESHOLD_PAIRS: List[Tuple[float, float]] = [
    (0.008, 0.01),
    (0.008, 0.02),
    (0.008, 0.03),
    (0.008, 0.04),
    (0.008, 0.05),
    (0.008, 0.06),
    (0.008, 0.07),
    (0.008, 0.08),
    (0.008, 0.09),
    (0.008, 0.10),
    (0.008, 0.11),
    (0.008, 0.12)
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("Optimizer")

def run_backtest_sync(config_path: str, data_dir: str, ticker: str, threshold_surplus: float, threshold_deficit: float, days: float = 0.125, quiet: bool = True):
    """Sync wrapper to run backtest in a separate process with asymmetric thresholds."""
    try:
        return asyncio.run(run_backtest(
            config_path=config_path,
            data_dir=data_dir,
            live_mode=False,
            ticker_override=ticker,
            threshold_surplus_override=threshold_surplus,
            threshold_deficit_override=threshold_deficit,
            days=days,
            quiet=quiet
        ))
    except Exception as e:
        return None

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Asymmetric threshold optimizer")
    parser.add_argument("--days", type=float, default=0.125,
                        help="Lookback window in days (default: 0.125 = 3 hours)")
    parser.add_argument("--debug-ticker", type=str, default=None,
                        help="Target specific ticker for deep logging to optimization_debug.log")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    all_tickers = config.get("tickers", [])
    if not all_tickers:
        logger.error("No tickers found in config.")
        return

    if args.debug_ticker:
        # Setup deep debug logging to file
        fh = logging.FileHandler('optimization_debug.log', mode='w')
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        fh.setFormatter(formatter)

        for log_name in ["Optimizer", "Backtest", "calculator"]:
            l = logging.getLogger(log_name)
            l.setLevel(logging.DEBUG)
            l.addHandler(fh)

        logger.info(f"DEBUG MODE ENABLED FOR {args.debug_ticker}. Writing to optimization_debug.log")
        tickers = [args.debug_ticker]
        all_tickers = tickers

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Filter out tickers with extreme price movements (>20% growth = SHORT liquidation risk)
    MAX_GROWTH_PCT = 20.0
    tickers = []
    skipped = []
    for ticker in all_tickers:
        try:
            import pandas as pd
            df = pd.read_feather(os.path.join(data_dir, f"{ticker}_live.feather"))
            start_price = df['close'].iloc[0]
            max_price = df['close'].max()
            growth_pct = (max_price / start_price - 1) * 100
            if growth_pct > MAX_GROWTH_PCT:
                skipped.append(f"{ticker} (+{growth_pct:.1f}%)")
            else:
                tickers.append(ticker)
        except FileNotFoundError:
            skipped.append(f"{ticker} (no data)")
    
    if skipped:
        logger.warning(f"Skipped {len(skipped)} tickers with >{MAX_GROWTH_PCT}% growth or no data: {', '.join(skipped)}")
    
    if not tickers:
        logger.error("No valid tickers remaining after filtering.")
        return
    
    logger.info(f"Optimizing {len(tickers)} tickers with {args.days:.3f} days lookback")
    logger.info(f"Threshold pairs to test (surplus, deficit): {len(THRESHOLD_PAIRS)}")

    # Results: (surplus, deficit) -> {avg_profit, avg_cycles}
    results_map: Dict[Tuple[float, float], Dict[str, float]] = {}

    if args.debug_ticker:
        # Single-threaded execution for clean logs
        for t_surplus, t_deficit in THRESHOLD_PAIRS:
            logger.info(f"--- DEBUG RUN: surplus={t_surplus:.3f}, deficit={t_deficit:.3f} ---")
            res = await run_backtest(
                config_path=config_path,
                data_dir=data_dir,
                live_mode=False,
                ticker_override=args.debug_ticker,
                threshold_surplus_override=t_surplus,
                threshold_deficit_override=t_deficit,
                days=args.days,
                quiet=False
            )
            if res:
                results_map[(t_surplus, t_deficit)] = {
                    "profit": res["profit_pct"],
                    "cycles": res["cycles"],
                    "max_dd": res["max_dd_pct"]
                }
                logger.info(f"  Result -> Profit: {res['profit_pct']:+.4f}% | Cycles: {res['cycles']} | Max DD: {res['max_dd_pct']:.2f}%")

        logger.info("Debug run complete. Check optimization_debug.log.")
        sys.exit(0)
    else:
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            loop = asyncio.get_running_loop()

            for t_surplus, t_deficit in THRESHOLD_PAIRS:
                logger.info(f"Testing thresholds: surplus={t_surplus:.3f}, deficit={t_deficit:.3f}...")

                tasks = [
                    loop.run_in_executor(executor, run_backtest_sync, config_path, data_dir, ticker, t_surplus, t_deficit, args.days)
                    for ticker in tickers
                ]

                ticker_results = [res for res in await asyncio.gather(*tasks) if res]

                if ticker_results:
                    avg_profit = sum(res["profit_pct"] for res in ticker_results) / len(ticker_results)
                    avg_cycles = sum(res["cycles"] for res in ticker_results) / len(ticker_results)
                    max_dd = max(res['max_dd_pct'] for res in ticker_results)

                    results_map[(t_surplus, t_deficit)] = {
                        "profit": avg_profit,
                        "cycles": avg_cycles,
                        "max_dd": max_dd
                    }
                    logger.info(f"  Result -> Avg Profit: {avg_profit:+.4f}% | Avg Cycles: {avg_cycles:.1f} | Max DD: {max_dd:.2f}%")
                else:
                    logger.warning(f"  No valid results for thresholds ({t_surplus}, {t_deficit})")

    if not results_map:
        logger.error("Optimization failed: no results collected.")
        return

    # Selection Logic: 
    # 1. We MUST have rebalances. If avg_cycles < 1.0, it's a "static" threshold, not rebalancing.
    # 2. Among those with cycles >= 1, pick the one with highest profit.
    # 3. If NONE have cycles >= 1, pick the one with highest cycles (attempting to rebalance).
    
    valid_candidates = {t: v for t, v in results_map.items() if v["cycles"] >= 1.0}
    
    if valid_candidates:
        best_pair = max(valid_candidates, key=lambda k: valid_candidates[k]["profit"])
    else:
        # Fallback: find the one that rebalances even a little bit
        best_pair = max(results_map, key=lambda k: results_map[k]["cycles"])
        logger.warning("No threshold pair achieved avg cycles >= 1.0. Picking pair with maximum activity.")

    best_profit = results_map[best_pair]["profit"]
    best_cycles = results_map[best_pair]["cycles"]
    best_dd = results_map[best_pair]["max_dd"]
    
    logger.info("=" * 60)
    logger.info(f"ASYMMETRIC THRESHOLD OPTIMIZATION COMPLETE")
    logger.info(f"Best Thresholds: surplus={best_pair[0]:.4f}, deficit={best_pair[1]:.4f}")
    logger.info(f"Expected Avg Profit: {best_profit:+.4f}%")
    logger.info(f"Expected Avg Cycles: {best_cycles:.1f}")
    logger.info(f"Max Drawdown: {best_dd:.2f}%")
    logger.info("=" * 60)

    # Update configuration with asymmetric thresholds
    config["portfolios"][0]["rebalance_threshold_surplus"] = best_pair[0]
    config["portfolios"][0]["rebalance_threshold_deficit"] = best_pair[1]
    
    # Remove legacy single threshold if present
    if "rebalance_threshold" in config["portfolios"][0]:
        del config["portfolios"][0]["rebalance_threshold"]
    
    # Update ticker-specific thresholds with asymmetric dict format
    if "ticker_thresholds" in config["portfolios"][0]:
        for ticker in tickers:
            config["portfolios"][0]["ticker_thresholds"][ticker] = {
                "surplus": best_pair[0],
                "deficit": best_pair[1]
            }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Updated {config_path} with new asymmetric thresholds.")

if __name__ == "__main__":
    asyncio.run(main())
