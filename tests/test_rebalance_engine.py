import asyncio
import pytest
from decimal import Decimal
from prosperous_bot.rebalance_engine import RebalanceEngine, _subst_symbol

class DummyPortfolio:
    """Minimal Portfolio stub for unit tests"""
    def __init__(self, dist, nav=Decimal("1000")):
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
    def __init__(self, filled=True, price=Decimal("1.0"), commission=Decimal("0.0")):
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
        return DummyOrder(filled=True, price=Decimal("100.0"))

    async def get_order(self, _id):
        # already filled
        return DummyOrder(filled=True, price=Decimal("100.0"))

    async def cancel_order(self, _id):
        self.calls.append(("CANCEL", _id))

    async def market_order(self, symbol, side, qty):
        self.calls.append(("MKT", symbol, side, qty))
        return DummyOrder(filled=True, price=Decimal("101.0"))

@pytest.fixture
def mock_exchange_state():
    return {
        "free_usdt": Decimal("5000.0"),
        "positions": [
            {"symbol": "BTCUSDT", "size": Decimal("0.5"), "mark_price": Decimal("60000.0")},
            {"symbol": "ETHUSDT", "size": Decimal("10.0"), "mark_price": Decimal("3000.0")}
        ]
    }

def test_nav_calculation(mock_exchange_state):
    engine = RebalanceEngine(portfolio=None) # portfolio=None should be fine for calculate_nav if it doesn't use it
    nav = engine.calculate_nav(mock_exchange_state)

    # 5000.0 + (0.5 * 60000.0) + (10.0 * 3000.0) = 65000.0
    expected_nav = Decimal("65000.0")
    assert nav == expected_nav, f"NAV mismatch: expected {expected_nav}, got {nav}"

def test_virtual_leg_physical_quantity_simulation():
    engine = RebalanceEngine(portfolio=None)
    virtual_position_size = engine.simulate_virtual_leg(
        allocation_usdt=Decimal("1000.0"),
        entry_price=Decimal("500.0")
    )

    expected_size = Decimal("2.0")
    assert virtual_position_size == expected_size, "Virtual leg size must be strictly equivalent to Decimal"

@pytest.mark.asyncio
async def test_build_orders_threshold():
    """RebalanceEngine should create orders only when diff exceeds threshold"""
    target = {"BTC_SPOT": Decimal("0.65"), "BTC_PERP_SHORT": Decimal("0.24"), "BTC_PERP_LONG": Decimal("0.11")}
    current = {"BTC_SPOT": Decimal("0.60"), "BTC_PERP_SHORT": Decimal("0.25"), "BTC_PERP_LONG": Decimal("0.15")}
    port = DummyPortfolio(current, nav=Decimal("1000"))
    engine = RebalanceEngine(
        portfolio=port,
        target_weights=target,
        spot_asset_symbol="BTCUSDT",
        futures_contract_symbol_base="BTCUSD_PERP",
        base_threshold_pct=Decimal("0.01"),
    )
    orders = await engine.build_orders(p_spot=Decimal("50000"), p_contract=Decimal("100"))
    # Spot diff 0.05 -> expect >=1 order
    assert orders, "No orders generated despite diff above threshold"
    # Validate structure
    for o in orders:
        assert {"symbol", "side", "qty"}.issubset(o.keys())
        assert o["qty"] >= Decimal("1")

@pytest.mark.asyncio
async def test_execute_post_only_success():
    """execute() should fill via PostOnly then update portfolio"""
    port = DummyPortfolio({}, nav=Decimal("1000"))
    exch = DummyExchange()
    engine = RebalanceEngine(
        portfolio=port,
        target_weights={},
        spot_asset_symbol="BTCUSDT",
        futures_contract_symbol_base="BTCUSD_PERP",
        exchange_client=exch,
    )
    orders = [dict(symbol="BTCUSDT", side="buy", qty=Decimal("1"), notional_usdt=Decimal("50"), asset_key="BTC_SPOT")]
    exec_log = await engine.execute(orders=orders, timeout_sec=1)
    assert exec_log and exec_log[0]["status"].startswith("filled")
    # Portfolio should record execution
    assert port.executions, "Portfolio.apply_execution was not called"

@pytest.mark.asyncio
async def test_rebalance_engine_init_no_target_weights():
    """Tests RebalanceEngine initialization without target_weights."""
    portfolio = DummyPortfolio({})
    params = {"main_asset_symbol": "BTC"}
    engine = RebalanceEngine(portfolio=portfolio, params=params)
    assert engine.target_weights == {}

@pytest.mark.asyncio
async def test_rebalance_engine_init_legacy_threshold():
    """Tests RebalanceEngine initialization with legacy threshold_pct."""
    portfolio = DummyPortfolio({})
    engine = RebalanceEngine(portfolio=portfolio, threshold_pct=Decimal("0.05"))
    assert engine.base_threshold_pct == Decimal("0.05")

def test_subst_symbol_recursive():
    """Tests _subst_symbol with nested lists and dicts."""
    obj = {
        "a": "{main_asset_symbol}_A",
        "b": ["{main_asset_symbol}_B", {"c": "{main_asset_symbol}_C"}]
    }
    result = _subst_symbol(obj, "BTC")
    assert result["a"] == "BTC_A"
    assert result["b"][0] == "BTC_B"
    assert result["b"][1]["c"] == "BTC_C"

def test_round_lot():
    """Tests the _round_lot static method."""
    assert RebalanceEngine._round_lot(Decimal("12.345"), Decimal("0.01")) == Decimal("12.34")
    assert RebalanceEngine._round_lot(Decimal("12.345"), Decimal("0.1")) == Decimal("12.3")
    assert RebalanceEngine._round_lot(Decimal("12.345"), Decimal("1")) == Decimal("12.0")

@pytest.mark.asyncio
async def test_build_orders_no_contract_price():
    """Tests build_orders when p_contract is None or zero."""
    target = {"BTC_PERP_LONG": Decimal("0.5")}
    current = {"BTC_PERP_LONG": Decimal("0.4")}
    port = DummyPortfolio(current, nav=Decimal("1000"))
    engine = RebalanceEngine(portfolio=port, target_weights=target, base_threshold_pct=Decimal("0.01"))
    
    # Test with p_contract = None
    orders_none = await engine.build_orders(p_spot=Decimal("50000"), p_contract=None)
    assert not orders_none

    # Test with p_contract = 0
    orders_zero = await engine.build_orders(p_spot=Decimal("50000"), p_contract=Decimal("0"))
    assert not orders_zero

@pytest.mark.asyncio
async def test_execute_no_exchange_client():
    """Tests that execute raises RuntimeError if exchange_client is not set."""
    engine = RebalanceEngine(portfolio=DummyPortfolio({}))
    with pytest.raises(RuntimeError):
        await engine.execute(orders=[])

class MockExchangeFallback(DummyExchange):
    async def post_only_limit(self, symbol, side, qty):
        self.calls.append(("POL", symbol, side, qty))
        # Simulate that the order is not filled immediately
        return DummyOrder(filled=False, price=Decimal("100.0"))

    async def get_order(self, _id):
        # Simulate that the order is still not filled
        return DummyOrder(filled=False, price=Decimal("100.0"))

@pytest.mark.asyncio
async def test_execute_fallback_to_market_order():
    """Tests if execute falls back to a market order if post-only fails."""
    port = DummyPortfolio({}, nav=Decimal("1000"))
    exch = MockExchangeFallback()
    engine = RebalanceEngine(portfolio=port, exchange_client=exch)
    orders = [dict(symbol="BTCUSDT", side="buy", qty=Decimal("1"), notional_usdt=Decimal("50"), asset_key="BTC_SPOT")]
    
    exec_log = await engine.execute(orders=orders, timeout_sec=1)
    
    assert any(call[0] == "MKT" for call in exch.calls), "Market order was not placed"
    assert any(call[0] == "CANCEL" for call in exch.calls), "Cancel order was not called"
    assert exec_log[0]["status"] == "filled_market"

class MockExchangeError(DummyExchange):
    async def post_only_limit(self, symbol, side, qty):
        raise ValueError("Exchange API Error")

@pytest.mark.asyncio
async def test_execute_error_handling():
    """Tests error handling during order execution."""
    port = DummyPortfolio({}, nav=Decimal("1000"))
    exch = MockExchangeError()
    engine = RebalanceEngine(portfolio=port, exchange_client=exch)
    orders = [dict(symbol="BTCUSDT", side="buy", qty=Decimal("1"), notional_usdt=Decimal("50"), asset_key="BTC_SPOT")]
    
    exec_log = await engine.execute(orders=orders)
    
    assert exec_log[0]["status"] == "error"

@pytest.mark.asyncio
async def test_rebalance_engine_init_legacy_target_weights(caplog):
    """Tests loading of legacy target_weights from params."""
    portfolio = DummyPortfolio({})
    params = {"target_weights": {"BTC_SPOT": Decimal("1.0")}}
    engine = RebalanceEngine(portfolio=portfolio, params=params)
    assert engine.target_weights == {"BTC_SPOT": Decimal("1.0")}
    assert "Using legacy 'target_weights' from params" in caplog.text

@pytest.mark.asyncio
async def test_rebalance_engine_init_direct_base_threshold(caplog):
    """Tests direct base_threshold_pct argument."""
    portfolio = DummyPortfolio({})
    engine = RebalanceEngine(portfolio=portfolio, base_threshold_pct=Decimal("0.02"))
    assert engine.base_threshold_pct == Decimal("0.02")
    assert "Using direct 'base_threshold_pct'" in caplog.text

class PortfolioNoNav(DummyPortfolio):
    def __init__(self, dist, nav=Decimal("1000")):
        super().__init__(dist, nav)
    
    # This portfolio does not have get_nav_usdt
    async def get_value_distribution_usdt(self, *_, **__):
        return self._dist

@pytest.mark.asyncio
async def test_build_orders_no_get_nav_usdt():
    """Tests build_orders with a portfolio that doesn't have get_nav_usdt."""
    target = {"BTC_SPOT": Decimal("0.6")}
    current = {"BTC_SPOT": Decimal("0.5")}
    # nav is implicitly calculated from the sum of values in the distribution
    port = PortfolioNoNav(current, nav=Decimal("1000"))
    engine = RebalanceEngine(portfolio=port, target_weights=target, base_threshold_pct=Decimal("0.01"))
    orders = await engine.build_orders(p_spot=Decimal("50000"))
    assert orders

class LegacyPortfolio(DummyPortfolio):
    async def get_value_distribution_usdt(self, p_spot, p_contract):
        # Old signature without leverage
        return self._dist

@pytest.mark.asyncio
async def test_build_orders_legacy_get_value_distribution():
    """Tests build_orders with a portfolio using the old get_value_distribution_usdt signature."""
    target = {"BTC_SPOT": Decimal("0.6")}
    current = {"BTC_SPOT": Decimal("0.5")}
    port = LegacyPortfolio(current, nav=Decimal("1000"))
    engine = RebalanceEngine(portfolio=port, target_weights=target, base_threshold_pct=Decimal("0.01"))
    orders = await engine.build_orders(p_spot=Decimal("50000"), p_contract=Decimal("100"))
    assert orders

@pytest.mark.asyncio
async def test_build_orders_unit_test_ctx():
    """Tests the is_unit_test_ctx logic."""
    target = {"BTC_SPOT": Decimal("0.6")}
    current = {"BTC_SPOT": Decimal("0.5")}
    port = DummyPortfolio(current, nav=Decimal("1000"))
    params = {"futures_leverage": Decimal("5.0")}
    engine = RebalanceEngine(portfolio=port, target_weights=target, base_threshold_pct=Decimal("0.01"), params=params)
    orders = await engine.build_orders(p_spot=Decimal("50000"))
    assert orders
    # In test context, qty is the delta_usdt
    assert orders[0]['qty'] == Decimal("100.0") # (0.6 - 0.5) * 1000
