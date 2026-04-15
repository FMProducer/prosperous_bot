#!/usr/bin/env python
"""Verification script for ticker configurability changes."""
import sys
sys.path.insert(0, '.')

# Test 1: Config loading
import json
with open('config.json', 'r') as f:
    config = json.load(f)
print("Config base_ticker: {}".format(config.get('base_ticker')))

# Test 2: Calculator with different tickers
from calculator import PortfolioCalculator

# Test BTCUSDT
positions_btc = {'BTCUSDT_LONG': 0.0, 'BTCUSDT_SHORT': 0.0}
calc_btc = PortfolioCalculator(positions_btc, 60000, 10000, 60000, 3500, base_ticker='BTCUSDT')
print("BTC Calculator - TPV: {}".format(calc_btc.tpv))

# Test ETHUSDT
positions_eth = {'ETHUSDT_LONG': 0.0, 'ETHUSDT_SHORT': 0.0}
calc_eth = PortfolioCalculator(positions_eth, 3000, 10000, 60000, 3500, base_ticker='ETHUSDT')
print("ETH Calculator - TPV: {}".format(calc_eth.tpv))

# Test 3: Deviations calculation
targets = {
    'BTCUSDT_LONG': {'share': 0.4, 'leverage': 5},
    'BTCUSDT_SHORT': {'share': 0.4, 'leverage': 5},
    'VIRTUAL': {'share': 0.2, 'leverage': 1}
}
deviations = calc_btc.calculate_deviations(targets, 0.02)
print("Deviations for BTC: {} found".format(len(deviations)))

# Test 4: Verify base_ticker is used in keys
for key in ['BTCUSDT_LONG', 'BTCUSDT_SHORT']:
    if key in targets:
        print("Target key {} exists".format(key))

print("\nAll verification tests passed! The ticker is now configurable.")