
import asyncio
import pytest
from prosperous_bot.rebalance_engine import RebalanceEngine

class DummyPortfolio:
    """Minimal Portfolio stub for unit tests"""
    def __init__(self, dist, nav=1000):
        self._dist = dist
        self._nav = nav
        self.executions = []

    async def get_nav_usdt(self, *_, **__):
        return self._nav

    async def get_value_distribution_usdt(self, *_, **__):
        return self._dist

    async def apply_execution(self, **kwargs):
        self.executions.append(kwargs)

class DummyOrder:
    def __init__(self, filled=True, price=1.0, commission=0.0):
        self.filled = filled
        self.price = price
        self.commission = commission
        self.id = "dummy"

class DummyExchange:
    """Mocks PostOnly-limit then Market order calls"""
    def __init__(self):
        self.calls = []

    async def post_only_limit(self, symbol, side, qty):
        self.calls.append(("POL", symbol, side, qty))
        return DummyOrder(filled=True, price=100.0)

    async def get_order(self, _id):
        # already filled
        return DummyOrder(filled=True, price=100.0)

    async def cancel_order(self, _id):
        self.calls.append(("CANCEL", _id))

    async def market_order(self, symbol, side, qty):
        self.calls.append(("MKT", symbol, side, qty))
        return DummyOrder(filled=True, price=101.0)

@pytest.mark.asyncio
async def test_build_orders_threshold():
    """RebalanceEngine should create orders only when diff exceeds threshold"""
    target = {"BTC_SPOT": 0.65, "BTC_SHORT5X": 0.24, "BTC_LONG5X": 0.11}
    current = {"BTC_SPOT": 0.60, "BTC_SHORT5X": 0.25, "BTC_LONG5X": 0.15}
    port = DummyPortfolio(current, nav=1000)
    engine = RebalanceEngine(
        portfolio=port,
        target_weights=target,
        spot_asset_symbol="BTCUSDT",
        futures_contract_symbol_base="BTCUSD_PERP",
        base_threshold_pct=0.01,
    )
    orders = await engine.build_orders(p_spot=50000, p_contract=100)
    # Spot diff 0.05 -> expect >=1 order
    assert orders, "No orders generated despite diff above threshold"
    # Validate structure
    for o in orders:
        assert {"symbol", "side", "qty"}.issubset(o.keys())
        assert o["qty"] >= 1

@pytest.mark.asyncio
async def test_execute_post_only_success():
    """execute() should fill via PostOnly then update portfolio"""
    port = DummyPortfolio({}, nav=1000)
    exch = DummyExchange()
    engine = RebalanceEngine(
        portfolio=port,
        target_weights={},
        spot_asset_symbol="BTCUSDT",
        futures_contract_symbol_base="BTCUSD_PERP",
        exchange_client=exch,
    )
    orders = [dict(symbol="BTCUSDT", side="buy", qty=1, notional_usdt=50, asset_key="BTC_SPOT")]
    exec_log = await engine.execute(orders=orders, timeout_sec=1)
    assert exec_log and exec_log[0]["status"].startswith("filled")
    # Portfolio should record execution
    assert port.executions, "Portfolio.apply_execution was not called"
