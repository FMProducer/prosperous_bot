
import pytest, gate_api
from prosperous_bot.rebalance_engine import RebalanceEngine

class MockPortfolio:
    def __init__(self, values: dict):
        self._initial_absolute_values = values
        self._nav = sum(values.values())

    async def get_nav_usdt(self, p_spot, p_contract=None, leverage=None): # Added parameters to match potential calls
        return self._nav

    async def get_value_distribution_usdt(self, p_spot, p_contract=None, leverage=None): # Added parameters
        if self._nav == 0:
            return {key: 0.0 for key in self._initial_absolute_values}
        return {
            key: value / self._nav
            for key, value in self._initial_absolute_values.items()
        }

    # get_positions_with_margin is not used by this specific test path in RebalanceEngine.build_orders
    # async def get_positions_with_margin(self):
    #     return [gate_api.Position(contract="BTC_USDT", size="4", margin="2000")]

@pytest.mark.asyncio
async def test_build_orders_pct_logic():
    portfolio = MockPortfolio({
        "BTC_SPOT": 6400,      # ~62.14%
        "BTC_PERP_SHORT": 2700,   # ~26.22%
        "BTC_PERP_LONG": 1200     # ~11.65%
    })

    target_weights = {
        "BTC_SPOT": 0.5,
        "BTC_PERP_SHORT": 0.3,
        "BTC_PERP_LONG": 0.2
    }
    # Pass params to RebalanceEngine, even if empty, to ensure self.params is initialized as a dict
    engine_params = {"futures_leverage": 5.0} # Example, RebalanceEngine uses this for leverage
    engine = RebalanceEngine(portfolio, target_weights=target_weights, threshold_pct=0.01, params=engine_params)
    # p_contract is intentionally omitted to test the path where futures orders might be skipped if p_contract is None
    orders = await engine.build_orders(p_spot=20000)

    assert isinstance(orders, list)
    assert len(orders) > 0
    for o in orders:
        assert "symbol" in o and "qty" in o
        assert o["qty"] >= 1 or o["symbol"] == "BTC_USDT"
