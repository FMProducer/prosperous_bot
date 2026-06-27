import pytest, gate_api
from prosperous_bot.rebalance_engine import RebalanceEngine
from math import isclose
from unittest.mock import MagicMock

class MockPortfolio:
    def __init__(self, values: dict):
        self._initial_absolute_values = values
        self._nav = sum(values.values())

    async def get_nav_usdt(self, p_spot, p_contract=None, leverage=None):
        return self._nav

    async def get_value_distribution_usdt(self, **kwargs):
        if self._nav == 0:
            return {key: 0.0 for key in self._initial_absolute_values}
        return {
            key: value / self._nav
            for key, value in self._initial_absolute_values.items()
        }

    async def apply_execution(self, **kwargs):
        pass

@pytest.mark.asyncio
async def test_build_orders_pct_logic():
    portfolio = MockPortfolio({
        "BTC_SPOT": 6400,
        "BTC_PERP_SHORT": 2700,
        "BTC_PERP_LONG": 1200
    })

    target_weights = {
        "BTC_SPOT": 0.5,
        "BTC_PERP_SHORT": 0.3,
        "BTC_PERP_LONG": 0.2
    }
    engine_params = {"futures_leverage": 5.0, "main_asset_symbol": "BTC"}
    engine = RebalanceEngine(portfolio, target_weights=target_weights, base_threshold_pct=0.01, params=engine_params)
    orders = await engine.build_orders(p_spot=20000, p_contract=20000)

    assert isinstance(orders, list)
    assert len(orders) > 0
    found_spot = any(o["asset_key"] == "BTC_SPOT" for o in orders)
    found_long = any(o["asset_key"] == "BTC_PERP_LONG" for o in orders)
    assert found_spot
    assert found_long

@pytest.mark.asyncio
async def test_rebalance_engine_init_params_none():
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    engine = RebalanceEngine(portfolio, params=None)
    assert engine.params == {}

@pytest.mark.asyncio
async def test_dynamic_threshold_logic():
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    engine = RebalanceEngine(portfolio, base_threshold_pct=0.01)
    assert isclose(engine._dynamic_threshold(0.01, 0.1), 0.02)
    assert isclose(engine._dynamic_threshold(0.01, 0.01), 0.01)

@pytest.mark.asyncio
async def test_rebalance_engine_execute_logic(mocker):
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    mock_exchange = MagicMock()
    mock_order_filled = MagicMock(id=1, filled=True, price=20000.0, commission=1.0)
    mock_exchange.post_only_limit = mocker.AsyncMock(return_value=mock_order_filled)
    mock_exchange.get_order = mocker.AsyncMock(return_value=mock_order_filled)

    engine = RebalanceEngine(portfolio, exchange_client=mock_exchange)
    orders = [{"symbol": "BTC_USDT", "side": "buy", "qty": 1.0}]
    results = await engine.execute(orders=orders, post_only=True)
    assert results[0]["status"] == "filled_limit"

    mock_order_unfilled = MagicMock(id=2, filled=False)
    mock_exchange.post_only_limit = mocker.AsyncMock(return_value=mock_order_unfilled)
    mock_exchange.get_order = mocker.AsyncMock(return_value=mock_order_unfilled)
    mock_exchange.cancel_order = mocker.AsyncMock()
    mock_market_order = MagicMock(price=20100.0, commission=2.0)
    mock_exchange.market_order = mocker.AsyncMock(return_value=mock_market_order)

    results = await engine.execute(orders=orders, post_only=True, timeout_sec=0)
    assert results[0]["status"] == "filled_market"
    assert results[0]["price_exec"] == 20100.0

@pytest.mark.asyncio
async def test_rebalance_engine_exceptions(mocker):
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    mock_exchange = MagicMock()
    mock_exchange.post_only_limit = mocker.AsyncMock(side_effect=Exception("API error"))
    engine = RebalanceEngine(portfolio, exchange_client=mock_exchange)
    orders = [{"symbol": "BTC_USDT", "side": "buy", "qty": 1.0}]
    results = await engine.execute(orders=orders, post_only=True)
    assert results[0]["status"] == "error"

@pytest.mark.asyncio
async def test_build_orders_p_contract_none():
    portfolio = MockPortfolio({"BTC_PERP_LONG": 1000})
    target_weights = {"BTC_PERP_LONG": 0.5}
    engine = RebalanceEngine(portfolio, target_weights=target_weights, params={"futures_leverage": 5.0, "main_asset_symbol": "BTC"})
    orders = await engine.build_orders(p_spot=20000, p_contract=None)
    assert len(orders) == 0

@pytest.mark.asyncio
async def test_rebalance_engine_init_logic():
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    engine = RebalanceEngine(portfolio, params={"main_asset_symbol": "ETH", "spot_asset_symbol": "{main_asset_symbol}_USDT"})
    assert engine.spot_asset_symbol == "ETH_USDT"

    engine = RebalanceEngine(portfolio, params={"base_threshold_pct": 0.07, "rebalance_threshold": 0.08})
    assert engine.base_threshold_pct == 0.08

@pytest.mark.asyncio
async def test_rebalance_engine_min_order_notional():
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    target_weights = {"BTC_SPOT": 0.99}
    engine = RebalanceEngine(portfolio, target_weights=target_weights, base_threshold_pct=0.001, params={"main_asset_symbol": "BTC", "min_order_notional_usdt": 15.0})
    orders = await engine.build_orders(p_spot=20000)
    assert len(orders) == 0
    engine = RebalanceEngine(portfolio, target_weights=target_weights, base_threshold_pct=0.001, params={"main_asset_symbol": "BTC", "min_order_notional_usdt": 5.0})
    orders = await engine.build_orders(p_spot=20000)
    assert len(orders) > 0

@pytest.mark.asyncio
async def test_rebalance_engine_debounce():
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    engine = RebalanceEngine(portfolio, target_weights={"BTC_SPOT": 0.5}, params={"main_asset_symbol": "BTC", "min_rebalance_interval_minutes": 10})
    orders = await engine.build_orders(p_spot=20000)
    assert len(orders) > 0
    orders = await engine.build_orders(p_spot=20000)
    assert len(orders) == 0

@pytest.mark.asyncio
async def test_rebalance_engine_threshold_skip():
    portfolio = MockPortfolio({"BTC_SPOT": 1000})
    engine = RebalanceEngine(portfolio, target_weights={"BTC_SPOT": 0.995}, threshold_pct=0.01, params={"main_asset_symbol": "BTC"})
    orders = await engine.build_orders(p_spot=20000)
    assert len(orders) == 0

@pytest.mark.asyncio
async def test_rebalance_engine_no_get_nav_usdt_attr():
    class SimplePortfolio:
        async def get_value_distribution_usdt(self, **kwargs):
            return {"BTC_SPOT": 1000.0}

    portfolio = SimplePortfolio()
    engine = RebalanceEngine(portfolio, target_weights={"BTC_SPOT": 0.5}, params={"main_asset_symbol": "BTC"})
    orders = await engine.build_orders(p_spot=20000)
    assert len(orders) > 0

@pytest.mark.asyncio
async def test_rebalance_engine_typeerror_dist():
    class TypeErrPortfolio:
        async def get_value_distribution_usdt(self, **kwargs):
             if 'leverage' in kwargs:
                 raise TypeError("Mocking first call failure")
             return {"BTC_SPOT": 1000.0}
    portfolio = TypeErrPortfolio()
    engine = RebalanceEngine(portfolio, target_weights={"BTC_SPOT": 0.5}, params={"main_asset_symbol": "BTC", "futures_leverage": 5.0})
    orders = await engine.build_orders(p_spot=20000, p_contract=20000)
    assert len(orders) > 0
