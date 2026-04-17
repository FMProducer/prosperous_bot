## Completed Tasks
- Implemented order prioritization in `main.py` and `backtest_rebalance.py`: SELL excess positions first, then BUY deficit positions.
- Enhanced logging in `main.py` and `backtest_rebalance.py`: Included TPV (USDT) and Shares (%) in "Rebalance needed" and "Cycle complete" logs. Removed "Cycle complete" logs for non-rebalancing cycles.
- Added PnL simulation in `main.py` for `paper_mode` to reflect equity changes dynamically and enable reinvestment.
- Updated `calculator.py` to store and return share percentages for logging.
- Implemented initial distribution logic in `main.py` if positions are empty upon startup.
- Implemented **Dynamic Threshold (ATR-based)** in `main.py` to automatically adjust rebalance threshold based on volatility.
- Updated `Rebalancer.md` with a detailed roadmap for future improvements (Limit Orders, Funding Arbitrage, Panic Mode).

## Next Steps
- Implement **Limit Orders (Post-Only)** in `executor.py` to reduce trading fees and slippage.
- Automate **Ticker Rotation** based on Funding Rate and Saw Factor from `rank_tickers.py`.
- Develop a **Volatility-based "Panic Mode"** to pause rebalancing during extreme spikes.
- Create a visualization tool for TPV and Reserve growth (HTML/CSV reporting).
