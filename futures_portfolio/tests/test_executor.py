import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from decimal import Decimal

from futures_portfolio.executor import PortfolioExecutor
from futures_portfolio.connector import BinanceConnector

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
    assert executor.round_quantity(Decimal("15.78"), Decimal("1.0")) == Decimal("15.0")
    assert executor.round_quantity(Decimal("0.123"), Decimal("0.0")) == Decimal("0.123")

@pytest.mark.asyncio
async def test_execute_market_order_success(mock_connector, executor):
    mock_connector.get_mark_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    mock_connector.get_order_trades.return_value = []
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.1"), "BUY", min_notional=Decimal("5.0"))
    assert result["status"] == "SUCCESS"
    assert result["executed_qty"] == 0.1

@pytest.mark.asyncio
async def test_execute_market_order_rounding_zero(mock_connector, executor):
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.0001"), "BUY", step_size=Decimal("0.1"))
    assert result["status"] == "NO_ORDER"

@pytest.mark.asyncio
async def test_execute_market_order_too_small(mock_connector, executor):
    mock_connector.get_mark_prices.return_value = {"BTCUSDT": 60000.0}
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.00001"), "BUY", min_notional=Decimal("6.0"))
    assert result["status"] == "SKIPPED"

@pytest.mark.asyncio
async def test_execute_market_order_price_fetch_error(mock_connector, executor):
    mock_connector.get_mark_prices.side_effect = Exception("Price error")
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    mock_connector.get_order_trades.return_value = []
    result = await executor.execute_market_order("BTCUSDT", Decimal("0.1"), "BUY", price=Decimal("60000.0"))
    assert result["status"] == "SUCCESS"

@pytest.mark.asyncio
async def test_execute_market_order_api_error(mock_connector, executor):
    mock_connector.get_mark_prices.return_value = {"BTCUSDT": 60000.0}
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
    mock_connector.get_order_trades.return_value = []
    result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.1"), "BUY", offset_pct=Decimal("0.0"))
    assert result["status"] == "SUCCESS_LIMIT"
    assert result["executed_qty"] == 0.1

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
    mock_connector.get_mark_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    mock_connector.get_order_trades.return_value = []
    result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.1"), "BUY")
    assert result["status"] == "SUCCESS" # Falls back to market which returns SUCCESS

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_timeout(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    mock_connector.place_limit_maker_order.return_value = {"orderId": 123}
    mock_connector.get_order_status.return_value = {"status": "NEW", "executedQty": "0"}
    mock_connector.get_mark_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000.0"})
    mock_connector.get_order_trades.return_value = []
    with patch("asyncio.sleep", return_value=None):
        result = await executor.execute_limit_with_fallback("BTCUSDT", Decimal("0.1"), "BUY", timeout_sec=2)
    assert result["status"] == "SUCCESS_FALLBACK"
