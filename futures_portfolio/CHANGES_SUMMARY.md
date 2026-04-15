# Summary of Changes for Configurable Ticker Support

## Problem
Changing just the `tickers` field in `config.json` was insufficient for replacing the base asset (e.g., from BTCUSDT to ETHUSDT). Many parts of the system had hardcoded references to BTCUSDT that needed to be made configurable.

## Solution
Made the base ticker configurable by:
1. Adding `base_ticker` field to `config.json`
2. Passing `base_ticker` parameter through all modules
3. Replacing hardcoded "BTCUSDT" strings with the configurable base_ticker

## Files Modified

### 1. config.json
- Added `base_ticker` field to support configurable base asset
- Default value: "BTCUSDT" (backward compatible)

### 2. main.py
- Reads `base_ticker` from config
- Passes it to PortfolioCalculator and other components
- Uses configurable ticker in all operations

### 3. calculator.py
- Added `base_ticker` parameter to `__init__`
- Replaced hardcoded "BTCUSDT" with `self.base_ticker` in:
  - Position value calculations
  - Share percentage calculations
  - Deviation calculations
  - Target keys (e.g., "BTCUSDT_LONG" → f"{base_ticker}_LONG")

### 4. connector.py
- Added `base_ticker` parameter to `__init__`
- Modified `get_spot_prices()` to use base_ticker for price lookups

### 5. executor.py
- Added `base_ticker` parameter to `__init__`
- Allows flexible order placement for any ticker

### 6. backtest_rebalance.py
- Added `base_ticker` variable from config
- Replaced hardcoded "BTCUSDT" with configurable ticker in:
  - Position initialization
  - PnL calculations
  - Order placement logic

### 7. tests/test_calculator.py
- Added `base_ticker_config` fixture
- Updated all tests to use configurable ticker
- Tests now work with any base ticker (BTCUSDT, ETHUSDT, etc.)

### 8. tests/test_executor.py
- Tests now work with any base ticker
- Maintains backward compatibility with BTCUSDT

## Backward Compatibility
All changes are backward compatible. If `base_ticker` is not specified in config.json, it defaults to "BTCUSDT", maintaining existing behavior.

## Usage Example
To switch from BTCUSDT to ETHUSDT:
1. Update `config.json`:
   ```json
   {
     "base_ticker": "ETHUSDT",
     "tickers": ["ETHUSDT"],
     ...
   }
   ```
2. Update portfolio targets to use ETHUSDT:
   ```json
   {
     "targets": {
       "ETHUSDT_LONG": {"share": 0.29, "leverage": 5},
       "ETHUSDT_SHORT": {"share": 0.36, "leverage": 5},
       "VIRTUAL": {"share": 0.35, "leverage": 1}
     }
   }
   ```

## Testing
All components have been verified to work with configurable tickers. The system now supports any trading pair, not just BTCUSDT.