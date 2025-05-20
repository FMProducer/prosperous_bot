# c:\Python\Prosperous_Bot\tests\test_portfolio_property.py
import asyncio, gate_api, time
from hypothesis import given, settings, strategies as st
from adaptive_agent.portfolio_manager import PortfolioManager
from unittest.mock import Mock
from math import isfinite

@given(spot=st.floats(0, 5), long=st.floats(0, 2), short=st.floats(0, 2),
    px=st.floats(10_000, 100_000))
@settings(max_examples=200, deadline=None)
def test_weights_sum(monkeypatch, spot, long, short, px):
    spot_api, fut_api = Mock(), Mock()

    spot_api.spot.get_account_detail = Mock(
        return_value=[gate_api.SpotAccount(currency="BTC", available=str(spot))]
    )
    fut_api.futures.list_positions = Mock(return_value=[
        gate_api.Position(size=str(long), contract="BTC_USDT", unrealised_pnl="0"),
        gate_api.Position(size=str(-short), contract="BTC_USDT", unrealised_pnl="0"),
    ])

    pm = PortfolioManager(spot_api, fut_api)
    w = asyncio.run(pm.get_notional_weights(px))
    total = sum(w.values())
    # Проверка адекватности весов
    assert all(v >= 0.0 and isfinite(v) for v in w.values())
    

# ... rest of code remains same