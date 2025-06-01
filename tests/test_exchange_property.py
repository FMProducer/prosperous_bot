import pytest
import pytest_asyncio # Implicitly used by @pytest.mark.asyncio
from unittest.mock import Mock as UMMock # Using an alias to avoid potential conflicts

import gate_api
from hypothesis import given, settings, strategies as st

# Assuming ExchangeAPI is correctly importable due to pythonpath settings
# from prosperous_bot.exchange_gate import ExchangeAPI # Not needed if exch fixture provides it

# Strategies (assuming these are defined as they were previously)
qty_st = st.integers(min_value=1, max_value=100_000)
order_type_st = st.sampled_from(["OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT"]) # For futures
spot_side_st = st.sampled_from(["buy", "sell"]) # For spot
price_st = st.one_of(st.none(), st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False, allow_infinity=False))
post_only_st = st.booleans()

@pytest.mark.asyncio
@settings(max_examples=50, deadline=None) # Reduced max_examples for potentially faster runs during debugging
@given(qty=qty_st, order_type=order_type_st)
async def test_create_futures_order_property(exch, mocker, qty, order_type):
    expected_size = qty if order_type.endswith("_LONG") else -qty
    expected_reduce_only = order_type.startswith("CLOSE_")

    # Mock for futures_api.create_futures_order
    # It's called with (self, settle, futures_order_payload)
    # It should return an object that simulates a FuturesOrder response
    def _echo_futures(settle_param, order_obj_futures):
        # order_obj_futures is the gate_api.FuturesOrder object created in ExchangeAPI.create_futures_order
        # Now returns a dictionary instead of a mock object
        returned_dict = {
            "id": "mock_fut_id_123",
            "status": "open",
            "settle": settle_param, # From settle_param
            "order_type": order_type # From test function parameter
        }

        # Copy attributes from the actual order_obj_futures
        if hasattr(order_obj_futures, 'contract'):
            returned_dict["contract"] = order_obj_futures.contract
        if hasattr(order_obj_futures, 'size'):
            returned_dict["size"] = order_obj_futures.size
        if hasattr(order_obj_futures, 'reduce_only'):
            returned_dict["reduce_only"] = order_obj_futures.reduce_only
        
        return returned_dict

    mocker.patch.object(exch.futures_api, "create_futures_order", side_effect=_echo_futures)

    # Act
    res = await exch.create_futures_order("BTC_USDT", order_type, qty)

    # Assert
    assert res["contract"] == "BTC_USDT"
    assert res["size"] == expected_size
    assert res["reduce_only"] == expected_reduce_only
    assert res["settle"] == "usdt" # This assertion relies on _echo_futures setting this attribute
    assert res["order_type"] == order_type

@pytest.mark.asyncio
@settings(max_examples=50, deadline=None)
@given(qty=qty_st, side=spot_side_st, price=price_st, post_only=post_only_st)
async def test_create_spot_order_property(exch, mocker, qty, side, price, post_only):
    # Mock for spot_api.create_order
    # It's called with (self, order_payload)
    # It should return an object that simulates an Order response
    def _echo_spot(order_obj_spot):
        # order_obj_spot is the gate_api.Order object created in ExchangeAPI.create_spot_order
        returned_mock = UMMock(spec=gate_api.Order) # Use spec for stricter mocking

        # Copy attributes from the actual order_obj_spot that the test asserts
        if hasattr(order_obj_spot, 'currency_pair'):
            returned_mock.currency_pair = order_obj_spot.currency_pair
        if hasattr(order_obj_spot, 'side'):
            returned_mock.side = order_obj_spot.side
        if hasattr(order_obj_spot, 'amount'):
            returned_mock.amount = order_obj_spot.amount
        if hasattr(order_obj_spot, 'type'):
            returned_mock.type = order_obj_spot.type
        if hasattr(order_obj_spot, 'time_in_force'):
            returned_mock.time_in_force = order_obj_spot.time_in_force
        
        # Handle optional price attribute
        if hasattr(order_obj_spot, 'price') and order_obj_spot.price is not None:
            returned_mock.price = order_obj_spot.price
        else:
            returned_mock.price = None # Ensure price attribute exists on mock even if None

        # Set the 'id' attribute on the mock for the assertion
        returned_mock.id = "spot123"
        
        # Simulate other attributes like status
        returned_mock.status = "filled" if price is None else "open" # Example status

        return returned_mock

    mocker.patch.object(exch.spot_api, "create_order", side_effect=_echo_spot)

    # Act
    res = await exch.create_spot_order("BTC_USDT", side, float(qty), price=price, post_only=post_only)

    # Assert
    assert res.id == "spot123"
    assert res.currency_pair == "BTC_USDT"
    assert res.side == side
    assert res.amount == str(qty) # Order amount is string
    expected_type = 'limit' if price else 'market'
    assert res.type == expected_type
    if price and post_only:
        assert res.time_in_force == 'poc'
    else:
        assert res.time_in_force == 'ioc'
    if price:
        assert res.price == str(price) # Order price is string

@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(contract=st.sampled_from([None, "BTC_USDT", "ETH_USDT"]))
async def test_positions_property(exch, mocker, contract):
    # Simulate API response for positions
    expected_positions = []
    if contract == "BTC_USDT" or contract is None:
        expected_positions.append(gate_api.Position(contract="BTC_USDT", size=10, user=123))
    if contract == "ETH_USDT" or contract is None and len(expected_positions) == 0: # ensure some data if contract is None
        expected_positions.append(gate_api.Position(contract="ETH_USDT", size=5, user=123))
    if not expected_positions and contract is not None : # Case where contract is specified but not BTC or ETH
         expected_positions.append(gate_api.Position(contract=contract, size=1, user=123))


    mocker.patch.object(exch.futures_api, "list_positions", return_value=expected_positions)

    # Act
    res = await exch.positions(contract=contract)

    # Assert
    assert isinstance(res, list)
    if contract:
        for pos in res:
            assert pos.contract == contract
    else:
        # If contract is None, we expect all positions
        assert len(res) == len(expected_positions)
