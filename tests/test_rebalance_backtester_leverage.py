import pytest
import pandas as pd
import copy
import tempfile
import os
from src.prosperous_bot.rebalance_backtester import run_backtest

# Minimal CSV market data
# Timestamp, Open, High, Low, Close, Volume
MARKET_DATA_CSV_CONTENT = """timestamp,open,high,low,close,volume
2023-01-01T00:00:00Z,100,105,95,100,10
2023-01-01T01:00:00Z,100,110,100,110,12 # Price increases
2023-01-01T02:00:00Z,110,110,100,100,15 # Price decreases
2023-01-01T03:00:00Z,100,105,95,90,13   # Price decreases further
"""

BASE_CONFIG = {
    "main_asset_symbol": "BTC",
    "initial_portfolio_value_usdt": 10000,
    "rebalance_threshold": 0.01, # Low threshold to ensure rebalancing occurs
    "target_weights_normal": {
        "BTC_SPOT": 0.4,
        "BTC_PERP_LONG": 0.3,
        "BTC_PERP_SHORT": 0.3
    },
    "circuit_breaker_config": {"enabled": False}, # Disable for simplicity
    "safe_mode_config": {
        "enabled": True, # Enable for one test case
        "metric_to_monitor": "margin_usage",
        "entry_threshold": 0.70, # Default
        "exit_threshold": 0.50, # Default
        "target_weights_safe": { # Define some safe weights
            "BTC_SPOT": 0.8,
            "BTC_PERP_LONG": 0.1,
            "BTC_PERP_SHORT": 0.1
        }
    },
    "data_settings": { # These paths won't be used if data_path is direct
        "csv_file_path": "dummy.csv",
        "signals_csv_path": "dummy_signals.csv"
    },
    "apply_signal_logic": False, # Disable signal logic for predictable rebalancing
    "min_rebalance_interval_minutes": 0, # Rebalance on every candle if needed
    "slippage_percent": 0, # No slippage for clearer P&L
    "commission_taker": 0, # No commission for clearer P&L
    "generate_reports_for_optimizer_trial": False,
    "report_path_prefix": "./reports_test_leverage/" # Avoid polluting main reports
}

@pytest.fixture
def market_data_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(MARKET_DATA_CSV_CONTENT)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)

def test_backtest_with_different_leverages(market_data_file):
    results = {}

    # Test Case 1: Leverage 5x (default behavior if not specified, but we specify for clarity)
    config_5x = copy.deepcopy(BASE_CONFIG)
    config_5x["futures_leverage"] = 5.0
    # Disable safe mode for these P&L comparison tests to isolate leverage effect on P&L
    config_5x["safe_mode_config"]["enabled"] = False
    results["5x"] = run_backtest(config_5x, data_path=market_data_file, is_optimizer_call=True)

    # Test Case 2: Leverage 10x
    config_10x = copy.deepcopy(BASE_CONFIG)
    config_10x["futures_leverage"] = 10.0
    config_10x["safe_mode_config"]["enabled"] = False
    results["10x"] = run_backtest(config_10x, data_path=market_data_file, is_optimizer_call=True)

    # Test Case 3: Leverage 1x
    config_1x = copy.deepcopy(BASE_CONFIG)
    config_1x["futures_leverage"] = 1.0
    config_1x["safe_mode_config"]["enabled"] = False
    results["1x"] = run_backtest(config_1x, data_path=market_data_file, is_optimizer_call=True)

    pnl_5x = results["5x"]["total_net_pnl_usdt"]
    pnl_10x = results["10x"]["total_net_pnl_usdt"]
    pnl_1x = results["1x"]["total_net_pnl_usdt"]

    # Assertions for P&L
    # With price movements: 100 -> 110 -> 100 -> 90
    # Initial rebalance at 100.
    # Long position: profits when price goes 100->110, loses 110->100, loses 100->90
    # Short position: loses when price goes 100->110, profits 110->100, profits 100->90
    # Overall, the market ends lower than it started (100 -> 90 after some up/down).
    # A larger leverage should amplify these changes.
    # The exact P&L is complex due to rebalancing, but we expect magnification of P&L magnitude.
    # P&L from futures will be (initial_value_futures / price_at_trade * leverage * price_change)
    # Given the price sequence (100 -> 110 -> 100 -> 90), futures positions will experience varied P&L.
    # Long: +10% change, then -9.09% change, then -10% change from last price.
    # Short: -10% change, then +9.09% change, then +10% change from last price.
    # The final P&L will depend on rebalancing points and amounts.
    # What we expect:
    # - PNL for 10x should have a larger magnitude than 5x.
    # - PNL for 1x should have the smallest magnitude for the futures component.
    # - Since the price ends down (100 -> 90), and we have both long and short,
    #   the short position should be net profitable, long net loss.
    #   The exact overall PNL is hard to predict without running, but the relationship should hold.

    # For this dataset, the price ends lower. A short position would be profitable, a long position not.
    # The net effect depends on the rebalancing.
    # Let's analyze the expected P&L change from the initial state (price 100).
    # Initial capital 10000. BTC_SPOT: 4000 (40 units), BTC_PERP_LONG: 3000, BTC_PERP_SHORT: 3000.
    # Price drops from 100 to 90 (a 10% drop).
    # Spot PNL: 40 units * (90-100) = -400
    # Long PNL (1x): 3000 * (-0.10) = -300
    # Short PNL (1x): 3000 * (+0.10) = +300
    # Total PNL (1x futures): -400 (spot) + 0 (futures) = -400 (ignoring rebalancing effects for simplicity)
    # Total PNL (5x futures): -400 (spot) + 5 * 0 = -400. This simplified view is wrong.
    # The PNL on the *value* of the position is Value * leverage * (price_change_percent)

    # Let's assume the first rebalance happens at price 100.
    # Portfolio: SPOT 4000 (40 BTC), LONG_VAL 3000, SHORT_VAL 3000.
    # Price moves 100 -> 110:
    #   SPOT: 40*110 = 4400 (+400)
    #   LONG_VAL (5x): 3000 + 3000 * 5 * (10/100) = 3000 + 1500 = 4500 (+1500)
    #   SHORT_VAL (5x): 3000 - 3000 * 5 * (10/100) = 3000 - 1500 = 1500 (-1500)
    #   Total at 110 before rebalance: 4400+4500+1500 = 10400. PNL = +400. (This is if rebalance doesn't happen till end)

    # This is still complex. The key is that the *change* in PNL due to the leveraged component
    # should scale with leverage.
    # PNL_total = PNL_spot + PNL_leveraged_futures
    # PNL_leveraged_futures_contrib_5x = pnl_5x - pnl_spot_component
    # PNL_leveraged_futures_contrib_10x = pnl_10x - pnl_spot_component
    # We expect PNL_leveraged_futures_contrib_10x to be roughly 2 * PNL_leveraged_futures_contrib_5x
    # And PNL_leveraged_futures_contrib_1x to be roughly 0.2 * PNL_leveraged_futures_contrib_5x

    # The spot PNL should be roughly the same in all runs if rebalancing doesn't drastically change spot holdings
    # due to futures P&L affecting total portfolio value.
    # For simplicity, we'll check if the PNLs are ordered as expected by leverage magnitude.
    # The specific PNL can be negative or positive.

    # If PNL from futures is positive, 10x > 5x > 1x.
    # If PNL from futures is negative, 10x < 5x < 1x (i.e. more negative).
    # The change from initial capital (0 PNL) is what we are looking at.

    # The provided market data results in a net loss for this strategy.
    # So, higher leverage should result in greater losses.
    assert pnl_10x < pnl_5x, f"10x leverage PNL ({pnl_10x}) should be less than 5x PNL ({pnl_5x}) for this losing scenario."
    assert pnl_5x < pnl_1x, f"5x leverage PNL ({pnl_5x}) should be less than 1x PNL ({pnl_1x}) for this losing scenario."

    # Also check that PNL for 1x is not excessively lossy - it should be somewhat protected.
    assert pnl_1x > -2000, f"1x PNL ({pnl_1x}) seems too low for a 10k portfolio with 1x leverage on futures."
                           # Max loss on spot is 4000 * (10/100) = 400. Max on futures (1x) similar. Sum ~ -800.
                           # This is a loose check.

def test_backtest_leverage_triggers_safe_mode(market_data_file):
    # Test Case 4: Leverage triggering Safe Mode
    # We need initial capital, positions, and price movement such that
    # margin_usage_ratio = ( (long_val/leverage) + (short_val/leverage) ) / nav
    # crosses the entry_threshold (0.70)

    config_safe_mode_low_leverage = copy.deepcopy(BASE_CONFIG)
    config_safe_mode_low_leverage["futures_leverage"] = 2.0 # Low leverage
    config_safe_mode_low_leverage["initial_portfolio_value_usdt"] = 1000 # Smaller capital to make margin usage higher
    config_safe_mode_low_leverage["target_weights_normal"] = { # Allocate significantly to futures
        "BTC_SPOT": 0.2,
        "BTC_PERP_LONG": 0.4,
        "BTC_PERP_SHORT": 0.4
    }
    config_safe_mode_low_leverage["safe_mode_config"]["enabled"] = True
    config_safe_mode_low_leverage["safe_mode_config"]["entry_threshold"] = 0.70
    # With 1000 USD capital, 400 LONG, 400 SHORT.
    # Margin with 2x leverage: (400/2) + (400/2) = 200 + 200 = 400.
    # Initial NAV is 1000. Margin Usage: 400 / 1000 = 0.4. This should NOT trigger safe mode.

    results_low_leverage = run_backtest(config_safe_mode_low_leverage, data_path=market_data_file, is_optimizer_call=True)

    config_safe_mode_high_leverage = copy.deepcopy(config_safe_mode_low_leverage)
    config_safe_mode_high_leverage["futures_leverage"] = 1.2 # Very low leverage denominator = high margin use
                                                            # Error in logic: low leverage means HIGH margin per unit of value
                                                            # Let's re-evaluate.
                                                            # Margin = Value / Leverage.
                                                            # If leverage is 1.2x: Margin = (400/1.2) + (400/1.2) = 333.33 + 333.33 = 666.66
                                                            # Margin usage: 666.66 / 1000 = 0.66. Still not > 0.70.

    # Let's try to make it higher. Target weights are 0.4 LONG, 0.4 SHORT.
    # So, 400 USDT for LONG, 400 USDT for SHORT.
    # For safe mode to trigger (threshold 0.7): ( (400/L) + (400/L) ) / NAV > 0.7
    # (800/L) / NAV > 0.7  => 800 / (L * NAV) > 0.7
    # Assuming NAV is close to initial_capital (1000) after first rebalance:
    # 800 / (L * 1000) > 0.7 => 0.8 / L > 0.7 => L < 0.8 / 0.7 => L < 1.14

    # So, a leverage of 1.1 should trigger it.
    # And leverage of 2.0 (as above) should not.

    config_safe_mode_high_leverage["futures_leverage"] = 1.1

    results_high_leverage = run_backtest(config_safe_mode_high_leverage, data_path=market_data_file, is_optimizer_call=True)

    assert results_low_leverage.get("num_safe_mode_entries", 0) == 0, \
        f"Safe mode should not be triggered with {config_safe_mode_low_leverage['futures_leverage']}x leverage. Entries: {results_low_leverage.get('num_safe_mode_entries', 0)}"

    assert results_high_leverage.get("num_safe_mode_entries", 0) > 0, \
        f"Safe mode SHOULD be triggered with {config_safe_mode_high_leverage['futures_leverage']}x leverage. Entries: {results_high_leverage.get('num_safe_mode_entries', 0)}"

    # Test with leverage exactly at a point where it might default due to effective_leverage_for_margin if leverage is 0
    config_zero_leverage_test = copy.deepcopy(BASE_CONFIG)
    config_zero_leverage_test["futures_leverage"] = 0.0
    config_zero_leverage_test["safe_mode_config"]["enabled"] = True # Keep safe mode enabled
    config_zero_leverage_test["initial_portfolio_value_usdt"] = 1000
    config_zero_leverage_test["target_weights_normal"] = { "BTC_SPOT": 0.2, "BTC_PERP_LONG": 0.4, "BTC_PERP_SHORT": 0.4 }

    # With leverage 0, effective_leverage_for_margin becomes 1e-9.
    # Margin usage = ( (400/1e-9) + (400/1e-9) ) / 1000. This will be extremely high.
    # So, safe mode should be triggered immediately if any futures positions are taken.
    results_zero_leverage = run_backtest(config_zero_leverage_test, data_path=market_data_file, is_optimizer_call=True)
    assert results_zero_leverage.get("num_safe_mode_entries", 0) > 0, \
        "Safe mode should be triggered with 0x leverage due to very high margin calculation."
    assert results_zero_leverage.get("time_steps_in_safe_mode", 0) > 0, \
        "Should spend time in safe mode with 0x leverage if positions are non-zero."

    # Also test P&L for 0 leverage: P&L from futures should be zero.
    # total_net_pnl for 0x leverage should mainly come from spot.
    # Price 100 -> 90. Spot is 0.2 * 1000 = 200 USDT (2 BTC at price 100).
    # Spot PNL = 2 * (90-100) = -20 USDT.
    # Commission and slippage are 0.
    # So, total_net_pnl_usdt should be close to -20.
    assert results_zero_leverage["total_net_pnl_usdt"] == pytest.approx(-20.0, abs=1.0), \
        f"PNL with 0x leverage should be spot PNL only. Got: {results_zero_leverage['total_net_pnl_usdt']}"

```
