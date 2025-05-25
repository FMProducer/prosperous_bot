
import pytest, gate_api
from prosperous_bot.rebalance_engine import RebalanceEngine

class MockPortfolio:
    def __init__(self, values: dict):
        self._values = values

    async def get_value_distribution_usdt(self, p_spot, p_contract=None):
        return self._values

    async def get_positions_with_margin(self):
        return [
            gate_api.Position(contract="BTC_USDT", size="4", margin="2000")
        ]

@pytest.mark.asyncio
async def test_build_orders_pct_logic():
    portfolio = MockPortfolio({
        "BTC_SPOT": 6400,      # ~62.14%
        "BTC_SHORT5X": 2700,   # ~26.22%
        "BTC_LONG5X": 1200     # ~11.65%
    })

    engine = RebalanceEngine(portfolio, threshold_pct=0.01)
    orders = await engine.build_orders(p_spot=20000)

    assert isinstance(orders, list)
    assert len(orders) > 0
    for o in orders:
        assert "symbol" in o and "qty" in o
        assert o["qty"] >= 1 or o["symbol"] == "BTC_USDT"
