# Ticker Migration Guide: BTCUSDT → Configurable Base Ticker

## Overview
This document describes the changes made to enable configurable base ticker symbols throughout the futures portfolio rebalancing system.

## Problem Statement
Previously, changing the base trading pair (e.g., from BTCUSDT to ETHUSDT) required modifying multiple hardcoded references throughout the codebase. This was error-prone and limited the system's flexibility.

## Solution Implemented
Made the base ticker symbol configurable through a new `base_ticker` field in `config.json`, which is propagated through all system components.

## Configuration Changes

### config.json
```json
{
  "api_key": "YOUR_API_KEY",
  "secret_key": "YOUR_SECRET_KEY",
  "testnet": true,
  "paper_mode": false,
  "portfolios": [...],
  "tickers": ["BTCUSDT"],
  "base_ticker": "BTCUSDT"  // NEW FIELD
}
```

## Code Changes

### 1. main.py
- Reads `base_ticker` from configuration
- Passes it to all components requiring ticker information
- Uses configurable ticker in all operations

### 2. calculator.py
**Changes:**
- Added `base_ticker` parameter to `__init__()`
- Replaced all hardcoded "BTCUSDT" references with `self.base_ticker`
- Modified position key generation to use configurable ticker

**Before:**
```python
long_notional = abs(self.positions.get("BTCUSDT_LONG", 0.0)) * self.price
```

**After:**
```python
long_notional = abs(self.positions.get(f"{self.base_ticker}_LONG", 0.0)) * self.price
```

### 3. connector.py
**Changes:**
- Added `base_ticker` parameter to `__init__()`
- Modified `get_spot_prices()` to use base_ticker for API calls

### 4. executor.py
**Changes:**
- Added `base_ticker` parameter to `__init__()`
- Allows flexible order placement for any trading pair

### 5. backtest_rebalance.py
**Changes:**
- Reads `base_ticker` from configuration
- Replaced hardcoded "BTCUSDT" in position initialization and PnL calculations

### 6. Test Files
**test_calculator.py:**
- Added `base_ticker_config` fixture
- All tests now use configurable ticker

**test_executor.py:**
- Tests updated to work with configurable ticker
- Maintains backward compatibility

## Migration Examples

### Example 1: Switch to ETHUSDT
1. Update `config.json`:
```json
{
  "base_ticker": "ETHUSDT",
  "tickers": ["ETHUSDT"],
  ...
}
```

2. Update portfolio targets:
```json
{
  "targets": {
    "ETHUSDT_LONG": {"share": 0.29, "leverage": 5},
    "ETHUSDT_SHORT": {"share": 0.36, "leverage": 5},
    "VIRTUAL": {"share": 0.35, "leverage": 1}
  }
}
```

### Example 2: Switch to SOLUSDT
```json
{
  "base_ticker": "SOLUSDT",
  "tickers": ["SOLUSDT"],
  ...
}
```

## Backward Compatibility
✓ All changes are backward compatible
✓ Default value for `base_ticker` is "BTCUSDT"
✓ Existing configurations continue to work without modification
✓ Test files maintain compatibility with both old and new patterns

## Benefits
1. **Flexibility**: Easily switch between different trading pairs
2. **Maintainability**: Single source of truth for ticker symbol
3. **Scalability**: System can now support multiple trading pairs
4. **Reduced Errors**: Eliminates risk of missing hardcoded references

## Testing
All components have been verified to work correctly with configurable tickers:
- Calculator functions correctly with different base tickers
- Connector properly fetches prices for configured ticker
- Executor places orders for any ticker
- Backtest rebalance works with configurable symbols
- All existing tests pass

## Future Enhancements
Potential improvements for future iterations:
1. Support for multiple base tickers simultaneously
2. Dynamic ticker switching at runtime
3. Configuration validation to ensure ticker consistency
4. Automated testing for multiple ticker scenarios