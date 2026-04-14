import pytest
from unittest.mock import Mock, patch

from futures_portfolio.executor import PortfolioExecutor
from futures_portfolio.connector import BinanceConnector


@pytest.fixture
def mock_connector():
    """Мок BinanceConnector с необходимыми методами."""
    connector = Mock(spec=BinanceConnector)
    # Mock futures_client
    mock_futures = Mock()
    mock_futures.futures_create_order = Mock()
    connector.futures_client = mock_futures
    return connector


def test_calculate_order_size_positive(mock_connector):
    """Тест расчёта ордера на открытие (BUY)."""
    executor = PortfolioExecutor(mock_connector)
    order_qty = executor.calculate_order_size(
        target_share=0.4,
        current_value=30000.0,
        total_value=100000.0,
        spot_price=60000.0,
    )
    # Ожидаем положительный результат
    assert order_qty > 0
    # Проверяем расчёт: (0.4*100000 - 30000) / 60000 = (40000 - 30000) / 60000 = 10000/60000 = 0.166666...
    assert pytest.approx(order_qty, 0.001) == 0.1667


def test_calculate_order_size_negative(mock_connector):
    """Тест расчёта ордера на закрытие (SELL)."""
    executor = PortfolioExecutor(mock_connector)
    order_qty = executor.calculate_order_size(
        target_share=0.2,
        current_value=40000.0,
        total_value=100000.0,
        spot_price=50000.0,
    )
    # Ожидаем отрицательный результат
    assert order_qty < 0
    # Проверяем расчёт: (0.2*100000 - 40000) / 50000 = (20000 - 40000) / 50000 = -20000/50000 = -0.4
    assert pytest.approx(order_qty, 0.0001) == -0.4


def test_round_quantity():
    """Тест округления количества."""
    executor = PortfolioExecutor(None)
    # BTCUSDT обычно имеет шаг 0.001
    assert executor.round_quantity(0.123456, 0.001) == 0.123
    # ETHUSDT обычно имеет шаг 0.01
    assert executor.round_quantity(0.123456, 0.01) == 0.12
    # Шаг 1.0
    assert executor.round_quantity(15.78, 1.0) == 16.0


@pytest.mark.asyncio
async def test_execute_market_order_with_rounding(mock_connector):
    """Тест отправки ордера с округлением."""
    executor = PortfolioExecutor(mock_connector)
    mock_result = {"orderId": 11223, "status": "FILLED"}
    mock_connector.futures_client.futures_create_order.return_value = mock_result

    # 0.12345678 -> 0.123 при шаге 0.001
    result = await executor.execute_market_order("BTCUSDT", 0.123456, "BUY", step_size=0.001)
    assert result["status"] == "SUCCESS"
    args, kwargs = mock_connector.futures_client.futures_create_order.call_args
    assert kwargs["quantity"] == 0.123


@pytest.mark.asyncio
async def test_execute_market_order_buy(mock_connector):
    """Тест отправки BUY ордера."""
    executor = PortfolioExecutor(mock_connector)
    # Подготавливаем мок-ответ
    mock_result = {"orderId": 12345, "status": "FILLED"}
    mock_connector.futures_client.futures_create_order.return_value = mock_result

    result = await executor.execute_market_order("BTCUSDT", 0.1, "BUY")
    assert result["status"] == "SUCCESS"
    mock_connector.futures_client.futures_create_order.assert_called_once()
    args, kwargs = mock_connector.futures_client.futures_create_order.call_args
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["side"] == "BUY"
    assert kwargs["quantity"] == 0.1
    assert kwargs["reduceOnly"] is False


@pytest.mark.asyncio
async def test_execute_market_order_sell(mock_connector):
    """Тест отправки SELL ордера."""
    executor = PortfolioExecutor(mock_connector)
    mock_result = {"orderId": 67890, "status": "FILLED"}
    mock_connector.futures_client.futures_create_order.return_value = mock_result

    result = await executor.execute_market_order("ETHUSDT", 0.05, "SELL")
    assert result["status"] == "SUCCESS"
    mock_connector.futures_client.futures_create_order.assert_called_once()
    args, kwargs = mock_connector.futures_client.futures_create_order.call_args
    assert kwargs["symbol"] == "ETHUSDT"
    assert kwargs["side"] == "SELL"
    assert kwargs["quantity"] == 0.05
    assert kwargs["reduceOnly"] is False


@pytest.mark.asyncio
async def test_execute_market_order_zero_quantity(mock_connector):
    """Тест, когда размер ордера равен нулю."""
    executor = PortfolioExecutor(mock_connector)
    result = await executor.execute_market_order("SOLUSDT", 0.0, "BUY")
    assert result["status"] == "NO_ORDER"
    mock_connector.futures_client.futures_create_order.assert_not_called()


@pytest.mark.asyncio
async def test_execute_market_order_api_error(mock_connector):
    """Тест обработки ошибки API."""
    executor = PortfolioExecutor(mock_connector)
    # Имитируем ошибку API
    mock_connector.futures_client.futures_create_order.side_effect = Exception("Test API error")
    
    result = await executor.execute_market_order("BTCUSDT", 0.1, "BUY")
    assert result["status"] == "ERROR"
    assert "Test API error" in result["message"]