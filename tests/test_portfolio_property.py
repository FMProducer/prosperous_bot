
import pytest, gate_api
from portfolio_manager import PortfolioManager
from unittest.mock import Mock
import asyncio
from hypothesis import given, strategies as st

@given(
    spot=st.floats(0.01, 1),
    short=st.integers(1, 10),
    long=st.integers(1, 10),
    price=st.floats(10000, 60000)
)
def test_value_distribution_valid(spot, short, long, price):
    spot_api = Mock()
    fut_api = Mock()
    spot_api.spot.get_account_detail = Mock(
        return_value=[gate_api.SpotAccount(currency="BTC", available=str(spot))]
    )
    fut_api.futures.list_positions = Mock(
        return_value=[
            gate_api.Position(size=str(long), contract="BTC_USDT", unrealised_pnl="0", margin="1000"),
            gate_api.Position(size=str(-short), contract="BTC_USDT", unrealised_pnl="0", margin="1000")
        ]
    )
    pm = PortfolioManager(spot_api, fut_api)
    values = asyncio.run(pm.get_value_distribution_usdt(price, 250))
    total = sum(values.values())
    assert all(v >= 0 for v in values.values())
    assert total > 0
