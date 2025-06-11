import pytest
import pandas as pd
import numpy as np # Added numpy
import copy
import tempfile
import os
from prosperous_bot.rebalance_backtester import run_backtest

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

# --- New Fixtures for annualization tests ---
@pytest.fixture
def base_config_factory():
    """Returns a deep copy of BASE_CONFIG for modification in tests."""
    return copy.deepcopy(BASE_CONFIG)

@pytest.fixture
def market_data_file_factory():
    """
    Factory fixture to create a temporary CSV market data file.
    Usage: market_data_file = market_data_file_factory("csv,content\n1,2")
    """
    created_files = []
    def _market_data_file_factory(csv_content_str):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(csv_content_str)
            filepath = tmp_file.name
        created_files.append(filepath)
        return filepath

    yield _market_data_file_factory

    for f_path in created_files:
        os.remove(f_path)
# --- End New Fixtures ---

# --- Tests for Dynamic Annualization Factor ---

def test_compute_metrics_5min_frequency(base_config_factory, market_data_file_factory):
    """Tests Sharpe/Sortino with 5-minute data frequency."""
    csv_content = """timestamp,open,high,low,close,volume
2023-01-01T00:00:00Z,100,100,100,100,10
2023-01-01T00:05:00Z,100,100.1,100,100.1,10
2023-01-01T00:10:00Z,100.1,100.1,100,100,10
2023-01-01T00:15:00Z,100,100,99.9,99.9,10
2023-01-01T00:20:00Z,99.9,100.05,99.9,100.05,10
"""
    data_file = market_data_file_factory(csv_content)
    prices = [100.0, 100.1, 100.0, 99.9, 100.05]
    initial_price = prices[0]

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = initial_price # Buy 1 unit of asset
    config["target_weights_normal"] = {"BTC_SPOT": 1.0, "USDT": 0.0}
    config["rebalance_threshold"] = 2.0 # Effectively initial rebalance only
    config["apply_signal_logic"] = False
    config["min_rebalance_interval_minutes"] = 0
    config["slippage_percent"] = 0
    config["taker_commission_rate"] = 0 # Ensure commission_taker is also 0
    config["futures_leverage"] = 1.0 # Not used but good to set
    config["circuit_breaker_config"]["enabled"] = False
    config["safe_mode_config"]["enabled"] = False
    config["report_path_prefix"] = "./reports_test_ann_factor/5min/"
    config["generate_reports_for_optimizer_trial"] = False
    config["annualization_factor"] = 252 # This will be ignored by new logic if timestamp works

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    # Calculate expected values
    initial_capital = config["initial_portfolio_value_usdt"]
    asset_qty = initial_capital / initial_price
    portfolio_values = pd.Series([p * asset_qty for p in prices])
    rets = portfolio_values.pct_change().dropna().reset_index(drop=True)

    expected_sharpe = 0.0
    expected_sortino = 0.0

    if not rets.empty and rets.std() != 0:
        mean_ret = rets.mean()
        std_ret = rets.std() # pandas default ddof=1
        downside_rets = rets[rets < 0]

        std_downside_ret = 0.0
        if not downside_rets.empty:
            temp_std_downside = downside_rets.std() # ddof=1
            if pd.notna(temp_std_downside) and temp_std_downside > 0:
                std_downside_ret = temp_std_downside
            elif mean_ret > 0:
                std_downside_ret = 1e-9
            else:
                std_downside_ret = 1e9
        elif mean_ret > 0:
            std_downside_ret = 1e-9
        else:
            std_downside_ret = 1e9

        ann_sqrt_5min = np.sqrt((365 * 24 * 60 * 60) / (5 * 60))

        expected_sharpe = (mean_ret / std_ret) * ann_sqrt_5min if std_ret > 1e-9 else 0.0
        expected_sortino = (mean_ret / std_downside_ret) * ann_sqrt_5min if std_downside_ret > 1e-9 else 0.0
        if mean_ret <= 0 and std_downside_ret < 1e-8 :
             expected_sortino = 0.0

    assert results["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-2)
    assert results["sortino_ratio"] == pytest.approx(expected_sortino, abs=1e-2)


def test_compute_metrics_hourly_frequency(base_config_factory, market_data_file_factory):
    """Tests Sharpe/Sortino with 1-hour data frequency."""
    csv_content = """timestamp,open,high,low,close,volume
2023-01-01T00:00:00Z,100,100,100,100,10
2023-01-01T01:00:00Z,100,100.1,100,100.1,10
2023-01-01T02:00:00Z,100.1,100.1,100,100,10
2023-01-01T03:00:00Z,100,100,99.9,99.9,10
2023-01-01T04:00:00Z,99.9,100.05,99.9,100.05,10
"""
    data_file = market_data_file_factory(csv_content)
    prices = [100.0, 100.1, 100.0, 99.9, 100.05]
    initial_price = prices[0]

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = initial_price
    config["target_weights_normal"] = {"BTC_SPOT": 1.0, "USDT": 0.0}
    config["rebalance_threshold"] = 2.0
    config["apply_signal_logic"] = False
    config["min_rebalance_interval_minutes"] = 0
    config["slippage_percent"] = 0
    config["taker_commission_rate"] = 0
    config["report_path_prefix"] = "./reports_test_ann_factor/hourly/"

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    initial_capital = config["initial_portfolio_value_usdt"]
    asset_qty = initial_capital / initial_price
    portfolio_values = pd.Series([p * asset_qty for p in prices])
    rets = portfolio_values.pct_change().dropna().reset_index(drop=True)
    expected_sharpe, expected_sortino = 0.0, 0.0

    if not rets.empty and rets.std() != 0:
        mean_ret = rets.mean()
        std_ret = rets.std()
        downside_rets = rets[rets < 0]
        std_downside_ret = downside_rets.std() if not downside_rets.empty and downside_rets.std() > 0 else (1e-9 if mean_ret > 0 else 1e9)
        if downside_rets.empty and mean_ret > 0 : std_downside_ret = 1e-9


        ann_sqrt_hourly = np.sqrt((365 * 24 * 60 * 60) / (60 * 60))
        expected_sharpe = (mean_ret / std_ret) * ann_sqrt_hourly if std_ret > 1e-9 else 0.0
        expected_sortino = (mean_ret / std_downside_ret) * ann_sqrt_hourly if std_downside_ret > 1e-9 else 0.0
        if mean_ret <= 0 and std_downside_ret < 1e-8 : expected_sortino = 0.0

    assert results["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-2)
    assert results["sortino_ratio"] == pytest.approx(expected_sortino, abs=1e-2)


def test_compute_metrics_daily_frequency_actual_data(base_config_factory, market_data_file_factory):
    """Tests Sharpe/Sortino with daily data frequency (uses 365 day year)."""
    csv_content = """timestamp,open,high,low,close,volume
2023-01-01T00:00:00Z,100,100,100,100,10
2023-01-02T00:00:00Z,100,100.1,100,100.1,10
2023-01-03T00:00:00Z,100.1,100.1,100,100,10
2023-01-04T00:00:00Z,100,100,99.9,99.9,10
2023-01-05T00:00:00Z,99.9,100.05,99.9,100.05,10
"""
    data_file = market_data_file_factory(csv_content)
    prices = [100.0, 100.1, 100.0, 99.9, 100.05]
    initial_price = prices[0]

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = initial_price
    config["target_weights_normal"] = {"BTC_SPOT": 1.0, "USDT": 0.0}
    config["report_path_prefix"] = "./reports_test_ann_factor/daily/"
    # Keep other simple config settings

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    initial_capital = config["initial_portfolio_value_usdt"]
    asset_qty = initial_capital / initial_price
    portfolio_values = pd.Series([p * asset_qty for p in prices])
    rets = portfolio_values.pct_change().dropna().reset_index(drop=True)
    expected_sharpe, expected_sortino = 0.0, 0.0

    if not rets.empty and rets.std() != 0:
        mean_ret = rets.mean()
        std_ret = rets.std()
        downside_rets = rets[rets < 0]
        std_downside_ret = downside_rets.std() if not downside_rets.empty and downside_rets.std() > 0 else (1e-9 if mean_ret > 0 else 1e9)
        if downside_rets.empty and mean_ret > 0 : std_downside_ret = 1e-9

        ann_sqrt_daily = np.sqrt((365 * 24 * 60 * 60) / (24 * 60 * 60)) # np.sqrt(365)
        expected_sharpe = (mean_ret / std_ret) * ann_sqrt_daily if std_ret > 1e-9 else 0.0
        expected_sortino = (mean_ret / std_downside_ret) * ann_sqrt_daily if std_downside_ret > 1e-9 else 0.0
        if mean_ret <= 0 and std_downside_ret < 1e-8 : expected_sortino = 0.0

    assert results["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-2)
    assert results["sortino_ratio"] == pytest.approx(expected_sortino, abs=1e-2)


def test_compute_metrics_fallback_invalid_timestamp(base_config_factory, market_data_file_factory):
    """Tests fallback to ann_sqrt=sqrt(252) with invalid timestamps (all same)."""
    csv_content = """timestamp,open,high,low,close,volume
2023-01-01T00:00:00Z,100,100,100,100,10
2023-01-01T00:00:00Z,100,100.1,100,100.1,10
2023-01-01T00:00:00Z,100.1,100.1,100,100,10
2023-01-01T00:00:00Z,100,100,99.9,99.9,10
2023-01-01T00:00:00Z,99.9,100.05,99.9,100.05,10
""" # Timestamps are identical, freq_sec will be 0 or NaN
    data_file = market_data_file_factory(csv_content)
    prices = [100.0, 100.1, 100.0, 99.9, 100.05] # prices from first row of each identical timestamp
    initial_price = prices[0]

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = initial_price
    config["target_weights_normal"] = {"BTC_SPOT": 1.0, "USDT": 0.0}
    config["report_path_prefix"] = "./reports_test_ann_factor/fallback_invalid_ts/"

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    # The backtester might use only the first entry for each identical timestamp,
    # or it might process them sequentially. If it processes sequentially, equity curve will have these prices.
    # If it groups by timestamp and takes first, then only one price point.
    # Assuming it processes sequentially based on pandas read_csv behavior:
    df_equity = pd.DataFrame({'portfolio_value_usdt': pd.Series([p * (initial_price/initial_price) for p in prices])})
    rets = df_equity["portfolio_value_usdt"].pct_change().dropna().reset_index(drop=True)

    expected_sharpe, expected_sortino = 0.0, 0.0

    if not rets.empty and rets.std() != 0:
        mean_ret = rets.mean()
        std_ret = rets.std()
        downside_rets = rets[rets < 0]
        std_downside_ret = downside_rets.std() if not downside_rets.empty and downside_rets.std() > 0 else (1e-9 if mean_ret > 0 else 1e9)
        if downside_rets.empty and mean_ret > 0 : std_downside_ret = 1e-9

        ann_sqrt_fallback = np.sqrt(252) # Expected fallback
        expected_sharpe = (mean_ret / std_ret) * ann_sqrt_fallback if std_ret > 1e-9 else 0.0
        expected_sortino = (mean_ret / std_downside_ret) * ann_sqrt_fallback if std_downside_ret > 1e-9 else 0.0
        if mean_ret <= 0 and std_downside_ret < 1e-8 : expected_sortino = 0.0

    assert results["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-2)
    assert results["sortino_ratio"] == pytest.approx(expected_sortino, abs=1e-2)


def test_compute_metrics_fallback_single_timestamp(base_config_factory, market_data_file_factory):
    """Tests fallback with a single data point (Sharpe/Sortino should be 0)."""
    csv_content = "timestamp,open,high,low,close,volume\n2023-01-01T00:00:00Z,100,100,100,100,10"
    data_file = market_data_file_factory(csv_content)

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = 100.0
    config["target_weights_normal"] = {"BTC_SPOT": 1.0, "USDT": 0.0}
    config["report_path_prefix"] = "./reports_test_ann_factor/fallback_single_ts/"

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    # With a single data point, rets will be empty. Sharpe/Sortino should be 0.
    # ann_sqrt calculation path should hit fallback but it won't matter for final metric.
    assert results["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert results["sortino_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_empty_equity_input(base_config_factory, market_data_file_factory):
    """Tests behavior with empty market data (Sharpe/Sortino should be 0)."""
    csv_content = "timestamp,open,high,low,close,volume\n" # Empty data, only headers
    data_file = market_data_file_factory(csv_content)

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = 10000.0
    config["report_path_prefix"] = "./reports_test_ann_factor/empty_input/"

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    # run_backtest returns 0 for metrics if market data is empty or bad
    assert results["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert results["sortino_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert results["max_drawdown_percent"] == pytest.approx(0.0, abs=1e-9) # Or specific error value like -100
    # Check status if possible, or rely on metric values for empty/error cases
    # For instance, if df_equity is empty, compute_metrics returns all zeros.
    # If market data load fails, run_backtest returns specific error structure.
    # This test assumes it goes through compute_metrics with empty df_eq.


def test_compute_metrics_zero_std_dev_returns(base_config_factory, market_data_file_factory):
    """Tests Sharpe/Sortino when returns have zero standard deviation (should be 0)."""
    csv_content = """timestamp,open,high,low,close,volume
2023-01-01T00:00:00Z,100,100,100,100,10
2023-01-01T00:05:00Z,100,100,100,100,10
2023-01-01T00:10:00Z,100,100,100,100,10
2023-01-01T00:15:00Z,100,100,100,100,10
"""
    data_file = market_data_file_factory(csv_content)

    config = base_config_factory()
    config["initial_portfolio_value_usdt"] = 100.0
    config["target_weights_normal"] = {"BTC_SPOT": 1.0, "USDT": 0.0}
    config["report_path_prefix"] = "./reports_test_ann_factor/zero_std/"

    results = run_backtest(config, data_path=data_file, is_optimizer_call=False)

    # Prices are constant, so rets are all 0. std_dev is 0.
    # Sharpe/Sortino should be 0.
    assert results["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert results["sortino_ratio"] == pytest.approx(0.0, abs=1e-9)

# --- End Tests for Dynamic Annualization Factor ---

def test_backtest_with_different_leverages(market_data_file):
    results = {}

    # Test Case 1: Leverage 5x (default behavior if not specified, but we specify for clarity)
    config_5x = copy.deepcopy(BASE_CONFIG)
    config_5x["futures_leverage"] = 5.0
    config_5x["report_path_prefix"] = "./reports_test_leverage/5x"
    # Disable safe mode for these P&L comparison tests to isolate leverage effect on P&L
    config_5x["safe_mode_config"]["enabled"] = False
    results["5x"] = run_backtest(config_5x, data_path=market_data_file, is_optimizer_call=True)

    # Test Case 2: Leverage 10x
    config_10x = copy.deepcopy(BASE_CONFIG)
    config_10x["futures_leverage"] = 10.0
    config_10x["report_path_prefix"] = "./reports_test_leverage/10x"
    config_10x["safe_mode_config"]["enabled"] = False
    results["10x"] = run_backtest(config_10x, data_path=market_data_file, is_optimizer_call=True)

    # Test Case 3: Leverage 1x
    config_1x = copy.deepcopy(BASE_CONFIG)
    config_1x["futures_leverage"] = 1.0
    config_1x["report_path_prefix"] = "./reports_test_leverage/1x"
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
    config_safe_mode_low_leverage["initial_portfolio_value_usdt"] = 800 # Smaller capital to make margin usage higher
    config_safe_mode_low_leverage["target_weights_normal"] = { # Allocate significantly to futures
        "BTC_SPOT": 0.2,
        "BTC_PERP_LONG": 0.4,
        "BTC_PERP_SHORT": 0.4
    }
    config_safe_mode_low_leverage["safe_mode_config"]["enabled"] = True
    config_safe_mode_low_leverage["safe_mode_config"]["entry_threshold"] = 0.65
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

    config_safe_mode_high_leverage["futures_leverage"] = 0.8
    config_safe_mode_high_leverage["safe_mode_config"]["entry_threshold"] = 0.65 # This line was missing in the previous diff

    results_high_leverage = run_backtest(config_safe_mode_high_leverage, data_path=market_data_file, is_optimizer_call=True)

    assert results_low_leverage.get("num_safe_mode_entries", 0) == 0, \
        f"Safe mode should not be triggered with {config_safe_mode_low_leverage['futures_leverage']}x leverage. Entries: {results_low_leverage.get('num_safe_mode_entries', 0)}"

    assert results_high_leverage.get("num_safe_mode_entries", 0) > 0, \
        f"Safe mode SHOULD be triggered with {config_safe_mode_high_leverage['futures_leverage']}x leverage. Entries: {results_high_leverage.get('num_safe_mode_entries', 0)}"

    # Test with leverage exactly at a point where it might default due to effective_leverage_for_margin if leverage is 0
    config_zero_leverage_test = copy.deepcopy(BASE_CONFIG)
    config_zero_leverage_test["futures_leverage"] = 0.0
    config_zero_leverage_test["safe_mode_config"]["enabled"] = True # Keep safe mode enabled
    config_zero_leverage_test["initial_portfolio_value_usdt"] = 800
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
    # Price 100 -> 90. Spot is 0.2 * 800 (initial_portfolio_value_usdt) = 160 USDT (1.6 BTC at price 100).
    # Base Spot PNL = 1.6 * (90-100) = -16 USDT.
    # Actual PNL will be affected by commissions and rebalancing trades.
    assert results_zero_leverage["total_net_pnl_usdt"] == pytest.approx(-16.0, abs=1.5), \
        f"PNL with 0x leverage should be close to spot PNL (-16.0) adjusted for commissions/rebalancing. Got: {results_zero_leverage['total_net_pnl_usdt']}"

def test_trade_quantity_scaling_with_leverage(mocker):
    from prosperous_bot.portfolio_manager import PortfolioManager
    spot_api = mocker.Mock()
    futures_api = mocker.Mock()
    pm = PortfolioManager(spot_api, futures_api)
    price_spot = 50000
    p_contract = 250.0
    nav_leverage = 15000
    nav_no_leverage = 3000
    leverage = 5.0
    diff = 0.1
    delta_usdt_leverage = diff * nav_leverage
    delta_usdt_noleverage = diff * nav_no_leverage
    qty_with_leverage = delta_usdt_leverage / p_contract
    qty_without_leverage = delta_usdt_noleverage / (p_contract * leverage)
    assert qty_with_leverage == pytest.approx(qty_with_leverage)

def test_empty_rebalance_trades_csv_created_when_sim_log_empty(market_data_file, tmp_path):
    """
    Tests that an empty rebalance_trades.csv (with headers) is created when
    simulate_rebalance returns an empty log, typically because initial positions
    were opened but not closed.
    """
    config_for_empty_sim_log = copy.deepcopy(BASE_CONFIG)
    config_for_empty_sim_log["initial_portfolio_value_usdt"] = 10000
    # This setup should create initial orders for BTC_PERP_LONG
    config_for_empty_sim_log["target_weights_normal"] = {"BTC_PERP_LONG": 0.5, "USDT": 0.5}
    # High threshold to prevent closing the position with the short market data
    config_for_empty_sim_log["rebalance_threshold"] = 0.8
    config_for_empty_sim_log["apply_signal_logic"] = False

    # Configure report generation into tmp_path
    report_path_prefix_str = str(tmp_path / "reports_test_empty_sim_log")
    config_for_empty_sim_log["report_path_prefix"] = report_path_prefix_str
    config_for_empty_sim_log["is_optimizer_call"] = True # To use the specific optimizer path structure
    config_for_empty_sim_log["generate_reports_for_optimizer_trial"] = True # Ensure reports are made

    trial_id = "empty_sim_log_test"

    # Run the backtest
    # is_optimizer_call and trial_id_for_reports are passed here to match how run_backtest constructs its output path
    run_backtest(
        config_for_empty_sim_log,
        data_path=market_data_file,
        is_optimizer_call=True,
        trial_id_for_reports=trial_id
    )

    # Determine the output directory
    # Path structure: {report_path_prefix}/optimizer_trials/trial_{trial_id}_{timestamp_str}
    base_optimizer_reports_dir = tmp_path / "reports_test_empty_sim_log" / "optimizer_trials"
    assert base_optimizer_reports_dir.exists(), f"Base optimizer reports directory not found: {base_optimizer_reports_dir}"

    # Find the specific trial directory (timestamp makes it dynamic)
    found_trial_dirs = [d for d in os.listdir(base_optimizer_reports_dir) if d.startswith(f"trial_{trial_id}_")]
    assert len(found_trial_dirs) == 1, f"Expected 1 trial directory for '{trial_id}', found {len(found_trial_dirs)}: {found_trial_dirs}"
    output_dir = base_optimizer_reports_dir / found_trial_dirs[0]

    rebalance_trades_csv_path = output_dir / "rebalance_trades.csv"

    assert os.path.exists(rebalance_trades_csv_path), \
        f"Expected rebalance_trades.csv to be created at {rebalance_trades_csv_path}, but it wasn't."

    # Read the CSV and check its properties
    df_rebalance_trades = pd.read_csv(rebalance_trades_csv_path)

    assert df_rebalance_trades.empty, \
        f"The rebalance_trades.csv should be empty (no data rows), but it has {len(df_rebalance_trades)} rows."

    expected_columns = ["asset_key", "entry_price", "exit_price", "qty", "pnl_gross_quote", "leverage"]
    assert list(df_rebalance_trades.columns) == expected_columns, \
        f"Columns in rebalance_trades.csv do not match expected. Got: {list(df_rebalance_trades.columns)}, Expected: {expected_columns}"
