import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
from decimal import Decimal

from futures_portfolio.core.executor import PortfolioExecutor
from futures_portfolio.core.connector import BinanceConnector

@pytest.fixture
def mock_connector():
    connector = Mock(spec=BinanceConnector)
    mock_futures = Mock()
    mock_futures.futures_create_order = Mock()
    connector.futures_client = mock_futures
    connector.get_futures_prices = AsyncMock()
    connector.get_order_book = AsyncMock()
    connector.place_limit_maker_order = AsyncMock()
    connector.get_order_status = AsyncMock()
    connector.cancel_order = AsyncMock()
    return connector

@pytest.fixture
def executor(mock_connector):
    return PortfolioExecutor(mock_connector)

def test_calculate_order_size(executor):
    qty = executor.calculate_order_size(0.4, 30000, 100000, 50000)
    assert qty == Decimal("0.2")

def test_round_quantity(executor):
    assert executor.round_quantity(Decimal("0.123456"), Decimal("0.001")) == Decimal("0.123")
    assert executor.round_quantity(Decimal("0.123456"), Decimal("0.01")) == Decimal("0.12")
    # Smart rounding uses max(step_size, 0.001) and ROUND_FLOOR
    assert executor.round_quantity(Decimal("15.78"), Decimal("1.0")) == Decimal("15")
    assert executor.round_quantity(Decimal("0.123"), Decimal("0.0")) == Decimal("0.123")

@pytest.mark.asyncio
async def test_execute_market_order_success(mock_connector, executor):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.1"), "BUY", min_notional=Decimal("5.0"))
    assert result["status"] == "SUCCESS"
    assert result["executed_qty"] == Decimal("0.1")

@pytest.mark.asyncio
async def test_execute_market_order_rounding_zero(mock_connector, executor):
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.0001"), "BUY", step_size=Decimal("0.1"))
    assert result["status"] == "NO_ORDER"

@pytest.mark.asyncio
async def test_execute_market_order_too_small(mock_connector, executor):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.00001"), "BUY", min_notional=Decimal("6.0"))
    assert result["status"] == "SKIPPED"

@pytest.mark.asyncio
async def test_execute_market_order_price_fetch_error(mock_connector, executor):
    mock_connector.get_futures_prices.side_effect = Exception("Price error")
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.1"), "BUY")
    assert result["status"] == "SUCCESS"

@pytest.mark.asyncio
async def test_execute_market_order_api_error(mock_connector, executor):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    with patch("asyncio.to_thread", side_effect=Exception("API Error")):
        result = await executor.execute_market_order("BTCUSDT", Decimal("0.1"), "BUY")
        assert result["status"] == "ERROR"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_success(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    mock_connector.place_limit_maker_order.return_value = {"orderId": 123}
    mock_connector.get_order_status.return_value = {"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000"}
    result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.1"), "BUY", offset_pct=Decimal("0.0"))
    assert result["status"] == "SUCCESS_LIMIT"
    assert result["executed_qty"] == Decimal("0.1")

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_too_small(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.00001"), "BUY", min_notional=Decimal("6.0"))
    assert result["status"] == "SKIPPED"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_error_then_market(mock_connector, executor):
    mock_connector.get_order_book.side_effect = Exception("Orderbook error")
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.1"), "BUY")
    # If limit fails, it falls back to market, which returns SUCCESS
    assert result["status"] == "SUCCESS"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_timeout(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    mock_connector.place_limit_maker_order.return_value = {"orderId": 123}
    mock_connector.get_order_status.return_value = {"status": "NEW", "executedQty": "0"}
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    with patch("asyncio.sleep", return_value=None):
        result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.1"), "BUY", timeout_sec=2)
    assert result["status"] == "SUCCESS_FALLBACK"

def test_get_limit_order_params(executor):
    config = {"limit_order_enabled": True, "limit_offset_pct": 0.5, "limit_timeout_sec": 60}
    enabled, offset, timeout = executor.get_limit_order_params(config)
    assert enabled is True
    assert offset == Decimal("0.5")
    assert timeout == 60

@pytest.mark.asyncio
async def test_execute_actions_surplus_first(executor, mock_connector):
    # Setup actions
    actions = [
        {"symbol": "ETHUSDT_LONG", "diff_usdt": 100.0, "is_reduction": False, "type": "ORDER"}, # Expansion
        {"symbol": "BTCUSDT_LONG", "diff_usdt": -100.0, "is_reduction": True, "type": "ORDER"} # Reduction
    ]
    
    # Track order of execution by mocking _execute_single_action
    execution_order = []
    
    async def mock_execute_single_action(action, price, paper_mode, portfolio_cfg, step_sizes, paper_state, mid_prices=None):
        execution_order.append(action["symbol"])
        return {"status": "SUCCESS", "symbol": action["symbol"]}

    with patch.object(PortfolioExecutor, "_execute_single_action", side_effect=mock_execute_single_action):
        await executor.execute_actions(actions, price=60000.0, paper_mode=True)
    
    # BTCUSDT (reduction) should be first
    assert execution_order == ["BTCUSDT_LONG", "ETHUSDT_LONG"]

@pytest.mark.asyncio
async def test_execute_single_action_paper_mode(executor):
    action = {"symbol": "BTCUSDT_LONG", "diff_usdt": 100.0, "type": "ORDER", "is_reduction": False}
    paper_state = {"long_entry_price": 50000.0}
    # price 60000. diff_usdt 100. qty = 100 / 60000 = 0.001666...
    # qty_rounded (step 0.001) = 0.001
    # reduce_only = False (side BUY, pos LONG)
    res = await executor._execute_single_action(
        action, price=60000.0, paper_mode=True, 
        portfolio_cfg={"min_notional_usdt": 6.0}, 
        step_sizes={"BTCUSDT": 0.001},
        paper_state=paper_state
    )
    assert res["status"] == "SUCCESS"
    assert res["qty"] == 0.001
    assert res["commission"] == pytest.approx(0.001 * 60000.0 * 0.0004)
    assert res["trade_pnl"] == 0.0 # expansion

    # Test reduce_only
    action_reduce = {"symbol": "BTCUSDT_LONG", "diff_usdt": -100.0, "type": "ORDER", "is_reduction": True}
    res = await executor._execute_single_action(
        action_reduce, price=60000.0, paper_mode=True, 
        portfolio_cfg={"min_notional_usdt": 6.0}, 
        step_sizes={"BTCUSDT": 0.001},
        paper_state=paper_state
    )
    assert res["reduce_only"] is True
    # trade_pnl = 0.001 * (60000 - 50000) = 10.0
    assert res["trade_pnl"] == 10.0

@pytest.mark.asyncio
async def test_execute_single_action_real_mode(executor, mock_connector):
    action = {"symbol": "BTCUSDT_LONG", "diff_usdt": 100.0, "type": "ORDER", "is_reduction": False}
    mock_connector.client = None
    mock_connector.verify_connection = AsyncMock()
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.001", "avgPrice": "60000.0"})
    
    with patch("asyncio.sleep", return_value=None), \
         patch.object(mock_connector, "get_order_trades", AsyncMock(return_value=[{"realizedPnl": "0.1", "commission": "0.05"}])):
        res = await executor._execute_single_action(
            action, price=60000.0, paper_mode=False, 
            portfolio_cfg={"min_notional_usdt": 6.0, "limit_order_enabled": False}, 
            step_sizes={"BTCUSDT": 0.001}
        )
    
    assert mock_connector.verify_connection.called
    assert res["status"] == "SUCCESS"
    assert res["trade_pnl"] == 0.1
    assert res["commission"] == 0.05

@pytest.mark.asyncio
async def test_execute_market_order_polling(executor, mock_connector):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    # First returns 0, then 0.1
    mock_connector.futures_client.futures_create_order = Mock(return_value={"orderId": 123, "executedQty": "0", "avgPrice": "0"})
    mock_connector.get_order_status.side_effect = [
        {"executedQty": "0", "avgPrice": "0"},
        {"executedQty": "0.1", "avgPrice": "60000.0"}
    ]
    
    with patch("asyncio.sleep", return_value=None), \
         patch.object(mock_connector, "get_order_trades", AsyncMock(return_value=[])):
        res = await executor.execute_market_order("BTCUSDT", 0.1, "BUY")
    
    assert res["status"] == "SUCCESS"
    assert res["executed_qty"] == Decimal("0.1")

@pytest.mark.asyncio
async def test_error_returns_contain_pnl_comm(executor, mock_connector):
    # Test market order error
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    with patch("asyncio.to_thread", side_effect=Exception("API FAIL")):
        res = await executor.execute_market_order("BTCUSDT", 0.1, "BUY")
        assert "trade_pnl" in res
        assert "commission" in res
    
    # Test _execute_single_action error in real mode
    action = {"symbol": "BTCUSDT_LONG", "diff_usdt": 100.0, "type": "ORDER", "is_reduction": False}
    mock_connector.client = MagicMock()
    with patch.object(executor, "execute_market_order", AsyncMock(return_value={"status": "ERROR", "trade_pnl": 0.0, "commission": 0.0})):
        res = await executor._execute_single_action(
            action, price=60000.0, paper_mode=False, 
            portfolio_cfg={"limit_order_enabled": False}, 
            step_sizes={}
        )
        assert res["status"] == "ERROR"
        assert res["trade_pnl"] == 0.0
        assert res["commission"] == 0.0

@pytest.mark.asyncio
async def test_execute_rebalance(executor, mock_connector):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    action = {"symbol": "BTCUSDT_LONG", "diff_usdt": 6000.0}
    res = await executor.execute_rebalance(action, price=60000.0, step_size=0.001, limit_order=False)
    assert res is True

@pytest.mark.asyncio
async def test_execute_single_action_virtual(executor):
    action = {"type": "VIRTUAL_ORDER", "diff_usdt": 100.0}
    res = await executor._execute_single_action(action, 60000.0, True, {}, {})
    assert res["type"] == "VIRTUAL_ORDER"
    assert res["status"] == "SUCCESS"

@pytest.mark.asyncio
async def test_execute_limit_immediate_fill(executor, mock_connector):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    mock_connector.place_limit_maker_order.return_value = {"orderId": 123}
    # Immediate fill
    mock_connector.get_order_status.return_value = {"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000"}
    
    with patch.object(mock_connector, "get_order_trades", AsyncMock(return_value=[])):
        res = await executor.execute_limit_with_fallback("BTCUSDT", 0.1, "BUY", timeout_sec=30)
    
    assert res["status"] == "SUCCESS_LIMIT"

@pytest.mark.asyncio
async def test_execute_single_action_skipped_dust(executor):
    action = {"symbol": "BTCUSDT_LONG", "diff_usdt": 1.0, "type": "ORDER"}
    res = await executor._execute_single_action(
        action, 60000.0, True, {"min_notional_usdt": 6.0}, {"BTCUSDT": 0.001}
    )
    assert res["status"] == "SKIPPED"
    assert "too small" in res["message"]

@pytest.mark.asyncio
async def test_execute_single_action_real_limit(executor, mock_connector):
    action = {"symbol": "BTCUSDT_LONG", "diff_usdt": 100.0, "type": "ORDER", "is_reduction": False}
    mock_connector.client = MagicMock()
    
    with patch.object(executor, "get_limit_order_params", return_value=(True, Decimal("0.2"), 30)), \
         patch.object(executor, "execute_limit_with_fallback", AsyncMock(return_value={"status": "SUCCESS_LIMIT", "executed_qty": Decimal("0.001"), "avg_price": Decimal("60000.0"), "realized_pnl": Decimal("0"), "commission": Decimal("0.02")})):
        res = await executor._execute_single_action(
            action, 60000.0, False, {}, {"BTCUSDT": 0.001}
        )
    assert res["status"] == "SUCCESS"
    assert res["commission"] == 0.02
