import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from futures_portfolio.connector import BinanceConnector, retry_on_network_error
from binance.exceptions import BinanceAPIException
import requests.exceptions

@pytest.fixture
def connector_setup():
    with patch("futures_portfolio.connector.AsyncClient") as mock_client:
        # Mock the instance created by AsyncClient()
        mock_instance = mock_client.return_value
        # Mock methods that are called on initialization or later
        mock_instance.futures_position_information = AsyncMock()
        mock_instance.futures_symbol_ticker = AsyncMock()
        mock_instance.futures_mark_price = AsyncMock()
        mock_instance.get_all_tickers = AsyncMock()
        mock_instance.futures_exchange_info = AsyncMock()
        mock_instance.futures_klines = AsyncMock()
        mock_instance.futures_account = AsyncMock()
        mock_instance.futures_get_position_mode = AsyncMock()
        mock_instance.futures_account_balance = AsyncMock()
        mock_instance.futures_change_leverage = AsyncMock()
        mock_instance.futures_change_margin_type = AsyncMock()
        mock_instance.futures_order_book = AsyncMock()
        mock_instance.futures_create_order = AsyncMock()
        mock_instance.futures_get_order = AsyncMock()
        mock_instance.futures_cancel_order = AsyncMock()

        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        yield conn, mock_instance

@pytest.mark.asyncio
async def test_get_positions(connector_setup):
    connector, mock_instance = connector_setup
    mock_positions = [
        {"symbol": "BTCUSDT", "positionAmt": "0.5", "entryPrice": "60000", "positionSide": "LONG"},
        {"symbol": "BTCUSDT", "positionAmt": "-0.2", "entryPrice": "61000", "positionSide": "SHORT"},
        {"symbol": "ETHUSDT", "positionAmt": "0", "entryPrice": "0", "positionSide": "BOTH"}
    ]
    mock_instance.futures_position_information.return_value = mock_positions
    positions = await connector.get_positions()
    assert "BTCUSDT_LONG" in positions
    assert positions["BTCUSDT_LONG"]["qty"] == 0.5
    assert "BTCUSDT_SHORT" in positions
    assert positions["BTCUSDT_SHORT"]["qty"] == -0.2
    assert "ETHUSDT" not in positions

@pytest.mark.asyncio
async def test_get_futures_prices(connector_setup):
    connector, mock_instance = connector_setup
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000"}]
    mock_instance.futures_symbol_ticker.return_value = mock_prices
    prices = await connector.get_futures_prices(["BTCUSDT"])
    assert prices["BTCUSDT"] == 60000.0

@pytest.mark.asyncio
async def test_get_futures_prices_none_tickers(connector_setup):
    connector, mock_instance = connector_setup
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000"}]
    mock_instance.futures_symbol_ticker.return_value = mock_prices
    prices = await connector.get_futures_prices(None)
    assert "BTCUSDT" in prices

@pytest.mark.asyncio
async def test_retry_decorator():
    mock_func = AsyncMock()
    mock_func.__name__ = "mock_func"
    mock_func.side_effect = [
        requests.exceptions.ConnectionError("Fail 1"),
        BinanceAPIException(Mock(status_code=500), 500, "Fail 2"),
        "Success"
    ]
    decorated = retry_on_network_error(retries=3, delay=0.01)(mock_func)
    result = await decorated()
    assert result == "Success"
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_retry_decorator_exhausted():
    mock_func = AsyncMock()
    mock_func.__name__ = "mock_func"
    mock_func.side_effect = requests.exceptions.ConnectionError("Fail")
    decorated = retry_on_network_error(retries=2, delay=0.01)(mock_func)
    with pytest.raises(requests.exceptions.ConnectionError):
        await decorated()
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_get_margin_ratio(connector_setup):
    connector, mock_instance = connector_setup
    mock_account = {"totalMarginBalance": "10000", "totalMaintMargin": "500", "availableBalance": "9500"}
    mock_instance.futures_account.return_value = mock_account
    info = await connector.get_margin_ratio()
    assert info["margin_ratio"] == 20.0

@pytest.mark.asyncio
async def test_get_hedge_mode(connector_setup):
    connector, mock_instance = connector_setup
    mock_instance.futures_get_position_mode.return_value = {"dualSidePosition": True}
    assert await connector.get_hedge_mode() is True

@pytest.mark.asyncio
async def test_get_free_balance(connector_setup):
    connector, mock_instance = connector_setup
    mock_balances = [{"asset": "USDT", "balance": "1000"}]
    mock_instance.futures_account_balance.return_value = mock_balances
    balance = await connector.get_free_balance()
    assert balance == 1000.0

@pytest.mark.asyncio
async def test_place_limit_order(connector_setup):
    connector, mock_instance = connector_setup
    mock_instance.futures_create_order.return_value = {"orderId": 123}
    res = await connector.place_limit_order("BTCUSDT", "BUY", 0.1, 60000, position_side="LONG")
    assert res["orderId"] == 123

@pytest.mark.asyncio
async def test_get_spot_prices(connector_setup):
    connector, mock_instance = connector_setup
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000"}]
    mock_instance.get_all_tickers.return_value = mock_prices
    prices = await connector.get_spot_prices(["BTCUSDT"])
    assert prices["BTCUSDT"] == 60000.0

@pytest.mark.asyncio
async def test_get_futures_klines(connector_setup):
    connector, mock_instance = connector_setup
    mock_klines = [["data"]]
    mock_instance.futures_klines.return_value = mock_klines
    klines = await connector.get_futures_klines("BTCUSDT", "1m")
    assert klines == mock_klines

@pytest.mark.asyncio
async def test_cancel_order(connector_setup):
    connector, mock_instance = connector_setup
    mock_instance.futures_cancel_order.return_value = {"status": "CANCELED"}
    res = await connector.cancel_order("BTCUSDT", 123)
    assert res["status"] == "CANCELED"

@pytest.mark.asyncio
async def test_place_limit_maker_order(connector_setup):
    connector, mock_instance = connector_setup
    mock_instance.futures_create_order.return_value = {"orderId": 456}
    res = await connector.place_limit_maker_order("BTCUSDT", "SELL", 0.1, 61000)
    assert res["orderId"] == 456

def test_connector_init_real_mode():
    with patch("futures_portfolio.connector.AsyncClient") as mock_client:
        conn = BinanceConnector(api_key="k", secret_key="s", testnet=False)
        assert conn.testnet is False
