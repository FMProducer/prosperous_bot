
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
    assert all(k in values for k in ("BTC_SPOT", "BTC_SHORT5X", "BTC_LONG5X"))
    assert all(isinstance(v, float) for v in values.values())
