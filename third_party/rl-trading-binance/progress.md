## Completed Tasks
- Implemented order prioritization in `main.py` and `backtest_rebalance.py`: SELL excess positions first, then BUY deficit positions.
- Enhanced logging in `main.py` and `backtest_rebalance.py`: Included TPV (USDT) and Shares (%) in "Rebalance needed" and "Cycle complete" logs. Removed "Cycle complete" logs for non-rebalancing cycles.
- Added PnL simulation in `main.py` for `paper_mode` to reflect equity changes dynamically and enable reinvestment.
- Updated `calculator.py` to store and return share percentages for logging.
- Implemented initial distribution logic in `main.py` if positions are empty upon startup.

## Next Steps
- Monitor bot performance with the updated logic.
- Consider adding more sophisticated error handling or recovery mechanisms.
- Investigate potential optimizations for order execution or rebalancing frequency.
