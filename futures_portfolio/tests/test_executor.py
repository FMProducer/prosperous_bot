import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

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
    assert qty == 0.2

def test_round_quantity(executor):
    assert executor.round_quantity(0.123456, 0.001) == 0.123
    assert executor.round_quantity(0.123456, 0.01) == 0.12
    assert executor.round_quantity(15.78, 1.0) == 16.0
    assert executor.round_quantity(0.123, 0.0) == 0.123

@pytest.mark.asyncio
async def test_execute_market_order_success(mock_connector, executor):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED"})
    result = await executor.execute_market_order("BTCUSDT", 0.1, "BUY", min_notional=5.0)
    assert result["status"] == "SUCCESS"

@pytest.mark.asyncio
async def test_execute_market_order_rounding_zero(mock_connector, executor):
    result = await executor.execute_market_order("BTCUSDT", 0.0001, "BUY", step_size=0.1)
    assert result["status"] == "NO_ORDER"

@pytest.mark.asyncio
async def test_execute_market_order_too_small(mock_connector, executor):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    result = await executor.execute_market_order("BTCUSDT", 0.00001, "BUY", min_notional=6.0)
    assert result["status"] == "SKIPPED"

@pytest.mark.asyncio
async def test_execute_market_order_price_fetch_error(mock_connector, executor):
    mock_connector.get_futures_prices.side_effect = Exception("Price error")
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED"})
    result = await executor.execute_market_order("BTCUSDT", 0.1, "BUY")
    assert result["status"] == "SUCCESS"

@pytest.mark.asyncio
async def test_execute_market_order_api_error(mock_connector, executor):
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    with patch("asyncio.to_thread", side_effect=Exception("API Error")):
        result = await executor.execute_market_order("BTCUSDT", 0.1, "BUY")
        assert result["status"] == "ERROR"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_success(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    mock_connector.place_limit_maker_order.return_value = {"orderId": 123}
    mock_connector.get_order_status.return_value = {"status": "FILLED", "executedQty": "0.1", "avgPrice": "60000"}
    result = await executor.execute_limit_with_fallback("BTCUSDT", 0.1, "BUY", offset_pct=0.0)
    assert result["status"] == "SUCCESS_LIMIT"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_too_small(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    result = await executor.execute_limit_with_fallback("BTCUSDT", 0.00001, "BUY", min_notional=6.0)
    assert result["status"] == "SKIPPED"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_error_then_market(mock_connector, executor):
    mock_connector.get_order_book.side_effect = Exception("Orderbook error")
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED"})
    result = await executor.execute_limit_with_fallback("BTCUSDT", 0.1, "BUY")
    assert result["status"] == "ERROR_FALLBACK"

@pytest.mark.asyncio
async def test_execute_limit_with_fallback_timeout(mock_connector, executor):
    mock_connector.get_order_book.return_value = {
        "bids": [["59990", "1"]],
        "asks": [["60010", "1"]]
    }
    mock_connector.place_limit_maker_order.return_value = {"orderId": 123}
    mock_connector.get_order_status.return_value = {"status": "NEW", "executedQty": "0"}
    mock_connector.get_futures_prices.return_value = {"BTCUSDT": 60000.0}
    mock_connector.futures_client.futures_create_order = Mock(return_value={"status": "FILLED"})
    with patch("asyncio.sleep", return_value=None):
        result = await executor.execute_limit_with_fallback("BTCUSDT", 0.1, "BUY", timeout_sec=2)
    assert result["status"] == "SUCCESS_FALLBACK"

def test_get_limit_order_params(executor):
    config = {"limit_order_enabled": True, "limit_offset_pct": 0.5, "limit_timeout_sec": 60}
    enabled, offset, timeout = executor.get_limit_order_params(config)
    assert enabled is True
    assert offset == 0.5
    assert timeout == 60
