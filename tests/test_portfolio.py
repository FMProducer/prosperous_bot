
import pytest, gate_api
from prosperous_bot.portfolio_manager import PortfolioManager
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_get_value_distribution_usdt(mocker):
    spot_api = mocker.Mock()
    fut_api = mocker.Mock()
    spot_api.spot.get_account_detail = mocker.Mock(return_value=[
        gate_api.SpotAccount(currency="BTC", available="0.1")
    ])
    fut_api.futures.list_positions = mocker.Mock(return_value=[
        gate_api.Position(size="0.02", contract="BTC_USDT", unrealised_pnl="50", margin="500"),
        gate_api.Position(size="-0.02", contract="BTC_USDT", unrealised_pnl="-20", margin="200")
    ])
    pm = PortfolioManager(spot_api, fut_api)
    values = await pm.get_value_distribution_usdt(p_spot=50000, p_contract=250)
    assert isinstance(values, dict)
    assert all(k in values for k in ("BTC_SPOT", "BTC_PERP_SHORT", "BTC_PERP_LONG"))
    assert all(isinstance(v, float) for v in values.values())


@pytest.mark.asyncio
async def test_get_value_distribution_with_leverage(mocker):
    spot_api_mock = mocker.Mock()
    futures_api_mock = mocker.Mock()

    # Mock spot account details
    # For spot: qty * p_spot
    mock_spot_btc_qty = 0.1
    spot_api_mock.spot.get_account_detail = mocker.Mock(return_value=[
        gate_api.SpotAccount(currency="BTC", available=str(mock_spot_btc_qty))
    ])

    # Mock futures positions
    # For futures: margin * leverage
    mock_long_margin = 100.0
    mock_short_margin = 50.0
    futures_api_mock.futures.list_positions = mocker.Mock(return_value=[
        gate_api.Position(size="0.01", contract="BTC_USDT", margin=str(mock_long_margin)),  # Long
        gate_api.Position(size="-0.005", contract="BTC_USDT", margin=str(mock_short_margin)) # Short
    ])

    pm = PortfolioManager(spot_api_mock, futures_api_mock, base_currency="BTC")

    test_cases = [
        # leverage, p_spot, p_contract (p_contract is not directly used for notional if margin exists)
        (5.0, 50000, 50000), # Default leverage
        (10.0, 50000, 50000), # Custom leverage
        (1.0, 60000, 60000),  # Leverage of 1
        (2.5, 40000, 40000), # Fractional leverage
    ]

    for leverage, p_spot, p_contract in test_cases:
        expected_spot_val = mock_spot_btc_qty * p_spot
        expected_long_val = mock_long_margin * leverage
        expected_short_val = mock_short_margin * leverage
        expected_total_value = expected_spot_val + expected_long_val + expected_short_val

        expected_distribution = {
            "BTC_SPOT": expected_spot_val / expected_total_value if expected_total_value else 0,
            "BTC_PERP_LONG": expected_long_val / expected_total_value if expected_total_value else 0,
            "BTC_PERP_SHORT": expected_short_val / expected_total_value if expected_total_value else 0,
        }
        if expected_total_value == 0: # Handle case for all zero values if it occurs
            expected_distribution = {"BTC_SPOT": 0.0, "BTC_PERP_LONG": 0.0, "BTC_PERP_SHORT": 0.0}


        # Test async version
        actual_values_async = await pm.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
        assert actual_values_async.keys() == expected_distribution.keys()
        for key in expected_distribution:
            assert actual_values_async[key] == pytest.approx(expected_distribution[key]), \
                f"Async mismatch for {key} with leverage {leverage}"

    # Test with no positions (only spot)
    futures_api_mock.futures.list_positions = mocker.Mock(return_value=[])
    leverage = 5.0
    p_spot = 50000
    p_contract = 50000
    expected_spot_val_only = mock_spot_btc_qty * p_spot
    expected_total_value_only = expected_spot_val_only
    expected_distribution_spot_only = {
        "BTC_SPOT": 1.0 if expected_total_value_only > 0 else 0.0,
        "BTC_PERP_LONG": 0.0,
        "BTC_PERP_SHORT": 0.0,
    }
    if expected_total_value_only == 0:
        expected_distribution_spot_only = {"BTC_SPOT": 0.0, "BTC_PERP_LONG": 0.0, "BTC_PERP_SHORT": 0.0}


    actual_values_async_spot_only = await pm.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
    assert actual_values_async_spot_only.keys() == expected_distribution_spot_only.keys()
    for key in expected_distribution_spot_only:
        assert actual_values_async_spot_only[key] == pytest.approx(expected_distribution_spot_only[key]), \
            f"Async mismatch for {key} (spot only)"

    actual_values_sync_spot_only = await pm.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
    assert actual_values_sync_spot_only.keys() == expected_distribution_spot_only.keys()
    for key in expected_distribution_spot_only:
        assert actual_values_sync_spot_only[key] == pytest.approx(expected_distribution_spot_only[key]), \
            f"Sync mismatch for {key} (spot only)"


    # Test with no spot or positions (all zero)
    # Async part
    spot_api_mock_all_zero_async = mocker.Mock()
    futures_api_mock_all_zero_async = mocker.Mock()
    spot_api_mock_all_zero_async.spot.get_account_detail = mocker.Mock(return_value=[
         gate_api.SpotAccount(currency="BTC", available="0.0")
    ])
    futures_api_mock_all_zero_async.futures.list_positions = mocker.Mock(return_value=[])
    pm_all_zero_async = PortfolioManager(spot_api_mock_all_zero_async, futures_api_mock_all_zero_async, base_currency="BTC")
    expected_distribution_all_zero = {"BTC_SPOT": 0.0, "BTC_PERP_LONG": 0.0, "BTC_PERP_SHORT": 0.0}

    actual_values_async_all_zero = await pm_all_zero_async.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
    assert actual_values_async_all_zero == expected_distribution_all_zero


# Separate test for the synchronous method to avoid event loop conflicts
def test_get_value_distribution_sync_with_leverage(mocker):
    spot_api_mock = mocker.Mock()
    futures_api_mock = mocker.Mock()

    mock_spot_btc_qty = 0.1
    spot_api_mock.spot.get_account_detail = mocker.Mock(return_value=[
        gate_api.SpotAccount(currency="BTC", available=str(mock_spot_btc_qty))
    ])

    mock_long_margin = 100.0
    mock_short_margin = 50.0
    futures_api_mock.futures.list_positions = mocker.Mock(return_value=[
        gate_api.Position(size="0.01", contract="BTC_USDT", margin=str(mock_long_margin)),
        gate_api.Position(size="-0.005", contract="BTC_USDT", margin=str(mock_short_margin))
    ])

    pm = PortfolioManager(spot_api_mock, futures_api_mock, base_currency="BTC")

    test_cases = [
        (5.0, 50000, 50000),
        (10.0, 50000, 50000),
        (1.0, 60000, 60000),
        (2.5, 40000, 40000),
    ]

    for leverage, p_spot, p_contract in test_cases:
        expected_spot_val = mock_spot_btc_qty * p_spot
        expected_long_val = mock_long_margin * leverage
        expected_short_val = mock_short_margin * leverage
        expected_total_value = expected_spot_val + expected_long_val + expected_short_val

        expected_distribution = {
            "BTC_SPOT": expected_spot_val / expected_total_value if expected_total_value else 0,
            "BTC_PERP_LONG": expected_long_val / expected_total_value if expected_total_value else 0,
            "BTC_PERP_SHORT": expected_short_val / expected_total_value if expected_total_value else 0,
        }
        if expected_total_value == 0:
            expected_distribution = {"BTC_SPOT": 0.0, "BTC_PERP_LONG": 0.0, "BTC_PERP_SHORT": 0.0}

        actual_values_sync = pm.get_value_distribution_sync(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
        assert actual_values_sync.keys() == expected_distribution.keys()
        for key in expected_distribution:
            assert actual_values_sync[key] == pytest.approx(expected_distribution[key]), \
                f"Sync mismatch for {key} with leverage {leverage}"

    # Test sync with no positions (only spot)
    futures_api_mock.futures.list_positions = mocker.Mock(return_value=[])
    leverage = 5.0
    p_spot = 50000
    p_contract = 50000 # Not used if no futures positions with margin
    expected_spot_val_only = mock_spot_btc_qty * p_spot
    expected_total_value_only = expected_spot_val_only
    expected_distribution_spot_only = {
        "BTC_SPOT": 1.0 if expected_total_value_only > 0 else 0.0,
        "BTC_PERP_LONG": 0.0,
        "BTC_PERP_SHORT": 0.0,
    }
    if expected_total_value_only == 0:
        expected_distribution_spot_only = {"BTC_SPOT": 0.0, "BTC_PERP_LONG": 0.0, "BTC_PERP_SHORT": 0.0}


    actual_values_sync_spot_only = pm.get_value_distribution_sync(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
    assert actual_values_sync_spot_only.keys() == expected_distribution_spot_only.keys()
    for key in expected_distribution_spot_only:
        assert actual_values_sync_spot_only[key] == pytest.approx(expected_distribution_spot_only[key]), \
            f"Sync mismatch for {key} (spot only)"

    # Test sync with no spot or positions (all zero)
    spot_api_mock_all_zero_sync = mocker.Mock()
    futures_api_mock_all_zero_sync = mocker.Mock()
    spot_api_mock_all_zero_sync.spot.get_account_detail = mocker.Mock(return_value=[
         gate_api.SpotAccount(currency="BTC", available="0.0")
    ])
    futures_api_mock_all_zero_sync.futures.list_positions = mocker.Mock(return_value=[]) # No positions
    pm_all_zero_sync = PortfolioManager(spot_api_mock_all_zero_sync, futures_api_mock_all_zero_sync, base_currency="BTC")
    expected_distribution_all_zero = {"BTC_SPOT": 0.0, "BTC_PERP_LONG": 0.0, "BTC_PERP_SHORT": 0.0}

    actual_values_sync_all_zero = pm_all_zero_sync.get_value_distribution_sync(p_spot=p_spot, p_contract=p_contract, leverage=leverage)
    assert actual_values_sync_all_zero == expected_distribution_all_zero
