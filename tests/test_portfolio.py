# c:\Python\Prosperous_Bot\tests\test_portfolio.py
import pytest, gate_api
from unittest.mock import Mock
from adaptive_agent.portfolio_manager import PortfolioManager


@pytest.mark.asyncio
async def test_nav_and_weights(monkeypatch):
    spot_api, fut_api = Mock(), Mock()

    # spot 0.1 BTC
    spot_api.spot.get_account_detail = Mock(return_value=[
        gate_api.SpotAccount(currency="BTC", available="0.1")
    ])
    # long 0.02 / short –0.02
    fut_api.futures.list_positions = Mock(return_value=[
        gate_api.Position(size="0.02", contract="BTC_USDT", unrealised_pnl="0"),
        gate_api.Position(size="-0.02", contract="BTC_USDT", unrealised_pnl="0"),
    ])

    pm = PortfolioManager(spot_api, fut_api)
    nav = await pm.nav_usd(50_000)
    assert nav == 0.1 * 50_000

    w = await pm.get_notional_weights(50_000)
    assert 1.3 <= sum(w.values()) <= 1.5
    assert w["BTC_SPOT"] > 0.99