import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import aiohttp
import json
from futures_portfolio.connector import BinanceConnector, retry_on_network_error
from binance.exceptions import BinanceAPIException
import requests.exceptions

def make_binance_exception(status_code, code, msg):
    response = Mock()
    data = {"code": code, "msg": msg}
    response.json.return_value = data
    return BinanceAPIException(response, status_code, json.dumps(data))

@pytest.fixture
def connector():
    with patch("futures_portfolio.connector.Client") as mock_client:
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        # Mocking Client and futures_client as if verify_connection was called
        conn.client = mock_client.return_value
        conn.futures_client = mock_client.return_value
        return conn

@pytest.mark.asyncio
async def test_get_positions(connector):
    mock_positions = [
        {"symbol": "BTCUSDT", "positionAmt": "0.5", "entryPrice": "60000", "positionSide": "LONG"},
        {"symbol": "BTCUSDT", "positionAmt": "-0.2", "entryPrice": "61000", "positionSide": "SHORT"},
        {"symbol": "ETHUSDT", "positionAmt": "0", "entryPrice": "0", "positionSide": "BOTH"}
    ]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_positions)):
        positions = await connector.get_positions()
    assert "BTCUSDT_LONG" in positions
    assert positions["BTCUSDT_LONG"]["qty"] == 0.5
    assert "BTCUSDT_SHORT" in positions
    assert positions["BTCUSDT_SHORT"]["qty"] == -0.2
    assert "ETHUSDT" not in positions

@pytest.mark.asyncio
async def test_get_futures_prices(connector):
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000"}]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_prices)):
        prices = await connector.get_futures_prices(["BTCUSDT"])
    assert prices["BTCUSDT"] == 60000.0

@pytest.mark.asyncio
async def test_get_futures_prices_dict_response(connector):
    mock_price = {"symbol": "BTCUSDT", "price": "60000"}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_price)):
        prices = await connector.get_futures_prices(["BTCUSDT"])
    assert prices["BTCUSDT"] == 60000.0

@pytest.mark.asyncio
async def test_get_futures_prices_none_tickers(connector):
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000"}]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_prices)):
        prices = await connector.get_futures_prices(None)
    assert "BTCUSDT" in prices

@pytest.mark.asyncio
async def test_retry_decorator():
    mock_func = AsyncMock()
    mock_func.__name__ = "mock_func"
    mock_func.side_effect = [
        requests.exceptions.ConnectionError("Fail 1"),
        make_binance_exception(500, -1000, "Fail 2"),
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
async def test_get_margin_ratio(connector):
    mock_account = {"totalMarginBalance": "10000", "totalMaintMargin": "500", "availableBalance": "9500"}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_account)):
        info = await connector.get_margin_ratio()
    assert info["margin_ratio"] == 20.0

@pytest.mark.asyncio
async def test_get_hedge_mode(connector):
    with patch("asyncio.to_thread", AsyncMock(return_value={"dualSidePosition": True})):
        assert await connector.get_hedge_mode() is True

@pytest.mark.asyncio
async def test_get_free_balance(connector):
    mock_balances = [{"asset": "USDT", "balance": "1000"}]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_balances)):
        balance = await connector.get_free_balance()
    assert balance == 1000.0

@pytest.mark.asyncio
async def test_place_limit_order(connector):
    with patch("asyncio.to_thread", AsyncMock(return_value={"orderId": 123})):
        res = await connector.place_limit_order("BTCUSDT", "BUY", 0.1, 60000, position_side="LONG")
    assert res["orderId"] == 123

@pytest.mark.asyncio
async def test_get_spot_prices(connector):
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000"}]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_prices)):
        prices = await connector.get_spot_prices(["BTCUSDT"])
    assert prices["BTCUSDT"] == 60000.0

@pytest.mark.asyncio
async def test_get_futures_klines(connector):
    mock_klines = [["data"]]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_klines)):
        klines = await connector.get_futures_klines("BTCUSDT", "1m")
    assert klines == mock_klines

@pytest.mark.asyncio
async def test_cancel_order(connector):
    with patch("asyncio.to_thread", AsyncMock(return_value={"status": "CANCELED"})):
        res = await connector.cancel_order("BTCUSDT", 123)
    assert res["status"] == "CANCELED"

@pytest.mark.asyncio
async def test_place_limit_maker_order(connector):
    with patch("asyncio.to_thread", AsyncMock(return_value={"orderId": 456})):
        res = await connector.place_limit_maker_order("BTCUSDT", "SELL", 0.1, 61000)
    assert res["orderId"] == 456

def test_connector_init_real_mode():
    with patch("futures_portfolio.connector.Client") as mock_client:
        conn = BinanceConnector(api_key="k", secret_key="s", testnet=False)
        assert conn.testnet is False

# --- NEW TESTS TO INCREASE COVERAGE TO >90% ---

@pytest.mark.asyncio
async def test_verify_connection_testnet_true():
    with patch("futures_portfolio.connector.Client") as mock_client:
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        assert conn.client is None

        await conn.verify_connection()

        mock_client.assert_called_once_with(
            "test_key", "test_secret", testnet=True,
            requests_params={'proxies': {'http': None, 'https': None}, 'timeout': 15}
        )
        assert conn.client == mock_client.return_value
        assert conn.futures_client == mock_client.return_value
        conn.futures_client.ping.assert_called_once()

@pytest.mark.asyncio
async def test_verify_connection_testnet_false():
    with patch("futures_portfolio.connector.Client") as mock_client:
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=False)
        await conn.verify_connection()

        mock_client.assert_called_once_with(
            "test_key", "test_secret", testnet=False,
            requests_params={'proxies': {'http': None, 'https': None}, 'timeout': 15}
        )
        client_instance = mock_client.return_value
        assert client_instance.API_URL == 'https://api1.binance.com/api'
        assert client_instance.FUTURES_URL == 'https://fapi.binance.com/fapi'
        assert client_instance.session.trust_env is False
        client_instance.ping.assert_called_once()

@pytest.mark.asyncio
async def test_verify_connection_already_initialized():
    with patch("futures_portfolio.connector.Client") as mock_client:
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        mock_client_inst = mock_client.return_value
        conn.client = mock_client_inst
        conn.futures_client = mock_client_inst

        await conn.verify_connection()
        mock_client.assert_not_called()
        mock_client_inst.ping.assert_called_once()

@pytest.mark.asyncio
async def test_verify_connection_init_binance_api_exception():
    with patch("futures_portfolio.connector.Client") as mock_client:
        mock_client.side_effect = make_binance_exception(400, -1000, "Invalid API Key")
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        with pytest.raises(BinanceAPIException):
            await conn.verify_connection()

@pytest.mark.asyncio
async def test_verify_connection_init_request_exception():
    with patch("futures_portfolio.connector.Client") as mock_client:
        mock_client.side_effect = requests.exceptions.RequestException("Connection error")
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        with pytest.raises(requests.exceptions.RequestException):
            await conn.verify_connection()

@pytest.mark.asyncio
async def test_verify_connection_ping_binance_api_exception():
    with patch("futures_portfolio.connector.Client") as mock_client:
        mock_client_inst = mock_client.return_value
        mock_client_inst.ping.side_effect = make_binance_exception(502, -1001, "Bad Gateway")
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        with pytest.raises(BinanceAPIException):
            await conn.verify_connection()

@pytest.mark.asyncio
async def test_verify_connection_ping_request_exception():
    with patch("futures_portfolio.connector.Client") as mock_client:
        mock_client_inst = mock_client.return_value
        mock_client_inst.ping.side_effect = requests.exceptions.RequestException("Ping failed")
        conn = BinanceConnector(api_key="test_key", secret_key="test_secret", testnet=True)
        with pytest.raises(requests.exceptions.RequestException):
            await conn.verify_connection()

@pytest.mark.asyncio
async def test_get_position_risk_empty_or_zero(connector):
    # Case 1: Empty positions
    with patch("asyncio.to_thread", AsyncMock(side_effect=[[], []])):
        res = await connector.get_position_risk()
        assert res == {}

    # Case 2: Zero quantities
    mock_positions = [
        {"symbol": "BTCUSDT", "positionAmt": "0.0", "positionSide": "LONG"},
        {"symbol": "ETHUSDT", "positionAmt": "0.0", "positionSide": "SHORT"}
    ]
    mock_mark_prices = [{"symbol": "BTCUSDT", "markPrice": "60000"}]
    with patch("asyncio.to_thread", AsyncMock(side_effect=[mock_positions, mock_mark_prices])):
        res = await connector.get_position_risk()
        assert res == {}

@pytest.mark.asyncio
async def test_get_position_risk_happy_path(connector):
    mock_positions = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.5",
            "positionSide": "LONG",
            "entryPrice": "60000",
            "liquidationPrice": "55000",
            "unrealizedProfit": "100.0",
            "isolatedMargin": "2000.0"
        },
        {
            "symbol": "ETHUSDT",
            "positionAmt": "-2.0",
            "positionSide": "SHORT",
            "entryPrice": "3000",
            "liquidationPrice": "3300",
            "unrealizedProfit": "-50.0",
            "isolatedMargin": "500.0"
        },
        {
            "symbol": "SOLUSDT",
            "positionAmt": "10.0",
            "positionSide": "BOTH",
            "entryPrice": "100",
            "liquidationPrice": "85",
            "unrealizedProfit": "150.0",
            "isolatedMargin": "100.0"
        },
        {
            "symbol": "ADAUSDT",
            "positionAmt": "-100.0",
            "positionSide": "BOTH",
            "entryPrice": "0.50",
            "liquidationPrice": "0.55",
            "unrealizedProfit": "-5.0",
            "isolatedMargin": "10.0"
        }
    ]

    mock_mark_prices_list = [
        {"symbol": "BTCUSDT", "markPrice": "60000"},
        {"symbol": "ETHUSDT", "markPrice": "3000"},
        {"symbol": "SOLUSDT", "markPrice": "100"},
        {"symbol": "ADAUSDT", "markPrice": "0.50"}
    ]
    with patch("asyncio.to_thread", AsyncMock(side_effect=[mock_positions, mock_mark_prices_list])):
        res = await connector.get_position_risk()

    assert "BTCUSDT_LONG" in res
    assert res["BTCUSDT_LONG"]["liq_price"] == 55000.0
    assert abs(res["BTCUSDT_LONG"]["distance_pct"] - 8.33333) < 1e-3
    assert res["BTCUSDT_LONG"]["unrealized_pnl"] == 100.0
    assert res["BTCUSDT_LONG"]["margin"] == 2000.0

    assert "ETHUSDT_SHORT" in res
    assert res["ETHUSDT_SHORT"]["liq_price"] == 3300.0
    assert abs(res["ETHUSDT_SHORT"]["distance_pct"] - 10.0) < 1e-3
    assert res["ETHUSDT_SHORT"]["unrealized_pnl"] == -50.0
    assert res["ETHUSDT_SHORT"]["margin"] == 500.0

    assert "SOLUSDT" in res
    assert res["SOLUSDT"]["liq_price"] == 85.0
    assert abs(res["SOLUSDT"]["distance_pct"] - 15.0) < 1e-3

    assert "ADAUSDT" in res
    assert res["ADAUSDT"]["liq_price"] == 0.55
    assert abs(res["ADAUSDT"]["distance_pct"] - 10.0) < 1e-3

@pytest.mark.asyncio
async def test_get_position_risk_edge_cases(connector):
    mock_positions = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.5",
            "positionSide": "LONG",
        }
    ]
    mock_mark_price_dict = {"symbol": "BTCUSDT", "markPrice": "60000"}
    with patch("asyncio.to_thread", AsyncMock(side_effect=[mock_positions, mock_mark_price_dict])):
        res = await connector.get_position_risk()

    assert "BTCUSDT_LONG" in res
    assert res["BTCUSDT_LONG"]["liq_price"] == 0.0
    assert res["BTCUSDT_LONG"]["distance_pct"] == 0.0
    assert res["BTCUSDT_LONG"]["unrealized_pnl"] == 0.0
    assert res["BTCUSDT_LONG"]["margin"] == 0.0

    mock_positions_zero_prices = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.5",
            "positionSide": "LONG",
            "entryPrice": "0.0",
            "liquidationPrice": "0.0",
        }
    ]
    with patch("asyncio.to_thread", AsyncMock(side_effect=[mock_positions_zero_prices, {"symbol": "BTCUSDT", "markPrice": "0.0"}])):
        res = await connector.get_position_risk()
    assert res["BTCUSDT_LONG"]["distance_pct"] == 0.0

    mock_positions_negative_distance = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.5",
            "positionSide": "LONG",
            "entryPrice": "60000",
            "liquidationPrice": "65000",
        }
    ]
    with patch("asyncio.to_thread", AsyncMock(side_effect=[mock_positions_negative_distance, {"symbol": "BTCUSDT", "markPrice": "60000"}])):
        res = await connector.get_position_risk()
    assert res["BTCUSDT_LONG"]["distance_pct"] == 0.0

@pytest.mark.asyncio
async def test_get_order_book(connector):
    mock_res = {"bids": [["50000", "1.0"]], "asks": [["50005", "2.0"]]}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_res)) as mock_to_thread:
        res = await connector.get_order_book("BTCUSDT", limit=10)
    assert res == mock_res
    mock_to_thread.assert_called_once_with(
        connector.futures_client.futures_order_book,
        symbol="BTCUSDT",
        limit=10
    )

@pytest.mark.asyncio
async def test_get_order_status(connector):
    mock_res = {"symbol": "BTCUSDT", "orderId": 12345, "status": "FILLED"}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_res)) as mock_to_thread:
        res = await connector.get_order_status("BTCUSDT", 12345)
    assert res == mock_res
    mock_to_thread.assert_called_once_with(
        connector.futures_client.futures_get_order,
        symbol="BTCUSDT",
        orderId=12345
    )

@pytest.mark.asyncio
async def test_get_order_trades(connector):
    mock_res = [{"id": 1, "price": "50000", "qty": "0.1"}]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_res)) as mock_to_thread:
        res = await connector.get_order_trades("BTCUSDT", 12345)
    assert res == mock_res
    mock_to_thread.assert_called_once_with(
        connector.futures_client.futures_account_trades,
        symbol="BTCUSDT",
        orderId=12345
    )

@pytest.mark.asyncio
async def test_set_leverage(connector):
    mock_res = {"symbol": "BTCUSDT", "leverage": 10, "maxNotionalValue": "1000000"}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_res)) as mock_to_thread:
        res = await connector.set_leverage("BTCUSDT", 10)
    assert res == mock_res
    mock_to_thread.assert_called_once_with(
        connector.futures_client.futures_change_leverage,
        symbol="BTCUSDT",
        leverage=10
    )

@pytest.mark.asyncio
async def test_set_margin_type_success(connector):
    mock_res = {"code": 200, "msg": "success"}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_res)) as mock_to_thread:
        res = await connector.set_margin_type("BTCUSDT", "ISOLATED")
    assert res == mock_res
    mock_to_thread.assert_called_once_with(
        connector.futures_client.futures_change_margin_type,
        symbol="BTCUSDT",
        marginType="ISOLATED"
    )

@pytest.mark.asyncio
async def test_set_margin_type_already_set(connector):
    exc = make_binance_exception(400, -4046, "No need to change margin type")
    with patch("asyncio.to_thread", AsyncMock(side_effect=exc)):
        res = await connector.set_margin_type("BTCUSDT", "ISOLATED")
    assert res is None

@pytest.mark.asyncio
async def test_set_margin_type_other_exception(connector):
    exc = make_binance_exception(400, -4000, "Some other error")
    with patch("asyncio.to_thread", AsyncMock(side_effect=exc)):
        with pytest.raises(BinanceAPIException):
            await connector.set_margin_type("BTCUSDT", "ISOLATED")

@pytest.mark.asyncio
async def test_get_bnb_balance(connector):
    connector.api_key = None
    assert await connector.get_bnb_balance() == 0.0

    connector.api_key = "YOUR_API_KEY"
    assert await connector.get_bnb_balance() == 0.0

    connector.api_key = "real_key"
    mock_balances = [
        {"asset": "USDT", "balance": "1000.0"},
        {"asset": "BNB", "balance": "5.5"}
    ]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_balances)):
        assert await connector.get_bnb_balance() == 5.5

    mock_balances_no_bnb = [
        {"asset": "USDT", "balance": "1000.0"}
    ]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_balances_no_bnb)):
        assert await connector.get_bnb_balance() == 0.0

@pytest.mark.asyncio
async def test_get_free_balance_edge_cases(connector):
    connector.api_key = None
    assert await connector.get_free_balance() == 10000.0

    connector.api_key = "YOUR_API_KEY"
    assert await connector.get_free_balance() == 10000.0

@pytest.mark.asyncio
async def test_get_exchange_info(connector):
    mock_res = {"symbols": [{"symbol": "BTCUSDT"}]}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_res)) as mock_to_thread:
        res = await connector.get_exchange_info()
    assert res == mock_res
    mock_to_thread.assert_called_once_with(connector.futures_client.futures_exchange_info)

@pytest.mark.asyncio
async def test_get_mark_prices_extra(connector):
    mock_price_dict = {"symbol": "BTCUSDT", "markPrice": "65000.0"}
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_price_dict)):
        res = await connector.get_mark_prices(None)
    assert res == {"BTCUSDT": 65000.0}

    mock_price_list = [
        {"symbol": "BTCUSDT", "markPrice": "65000.0"},
        {"symbol": "ETHUSDT", "markPrice": "3200.0"}
    ]
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_price_list)):
        res = await connector.get_mark_prices(["BTCUSDT"])
    assert res == {"BTCUSDT": 65000.0}

@pytest.mark.asyncio
async def test_get_spot_prices_extra(connector):
    mock_prices = [{"symbol": "BTCUSDT", "price": "60000.0"}]
    connector.base_ticker = "BTCUSDT"
    with patch("asyncio.to_thread", AsyncMock(return_value=mock_prices)):
        res = await connector.get_spot_prices(None)
    assert res == {"BTCUSDT": 60000.0}

    with patch("asyncio.to_thread", AsyncMock(return_value=mock_prices)):
        res = await connector.get_spot_prices(["BTCUSDT_LONG"])
    assert res == {"BTCUSDT_LONG": 60000.0}

@pytest.mark.asyncio
async def test_retry_on_network_error_requests_exceptions():
    mock_func = AsyncMock()
    mock_func.__name__ = "test_func"
    mock_func.side_effect = [
        requests.exceptions.RequestException("Request failed"),
        "Success"
    ]

    decorated = retry_on_network_error(retries=3, delay=1.5)(mock_func)

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await decorated()

    assert res == "Success"
    assert mock_func.call_count == 2
    mock_sleep.assert_called_once_with(1.5)

@pytest.mark.asyncio
async def test_retry_on_network_error_binance_server_error():
    mock_func = AsyncMock()
    mock_func.__name__ = "test_func"
    exc_500 = make_binance_exception(500, -1000, "Server Error")
    mock_func.side_effect = [exc_500, "Success"]

    decorated = retry_on_network_error(retries=3, delay=2.0)(mock_func)
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await decorated()
    assert res == "Success"
    assert mock_func.call_count == 2
    mock_sleep.assert_called_once_with(2.0)

    mock_func_429 = AsyncMock()
    mock_func_429.__name__ = "test_func_429"
    exc_429 = make_binance_exception(429, -1003, "Rate Limit")
    mock_func_429.side_effect = [exc_429, "Success"]

    decorated_429 = retry_on_network_error(retries=3, delay=2.0)(mock_func_429)
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await decorated_429()
    assert res == "Success"
    assert mock_func_429.call_count == 2
    mock_sleep.assert_called_once_with(2.0)

@pytest.mark.asyncio
async def test_retry_on_network_error_client_errors():
    for code in [-2019, -4003, -4164]:
        mock_func = AsyncMock()
        mock_func.__name__ = f"test_func_{code}"
        exc = make_binance_exception(400, code, f"Client Error {code}")
        mock_func.side_effect = exc

        decorated = retry_on_network_error(retries=3, delay=1.0)(mock_func)
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            with pytest.raises(BinanceAPIException) as exc_info:
                await decorated()
            assert exc_info.value.code == code
            assert mock_func.call_count == 1
            mock_sleep.assert_not_called()

@pytest.mark.asyncio
async def test_retry_on_network_error_exhausted_server_error():
    mock_func = AsyncMock()
    mock_func.__name__ = "test_func"
    exc_500 = make_binance_exception(500, -1000, "Server Error")
    mock_func.side_effect = exc_500

    decorated = retry_on_network_error(retries=3, delay=2.0)(mock_func)
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(BinanceAPIException):
            await decorated()
    assert mock_func.call_count == 3
    assert mock_sleep.call_count == 2

@pytest.mark.asyncio
async def test_retry_on_network_error_binance_none_status_code():
    mock_func = AsyncMock()
    mock_func.__name__ = "test_func"
    exc_none = make_binance_exception(None, -1000, "Unknown Error")
    mock_func.side_effect = [exc_none, "Success"]

    decorated = retry_on_network_error(retries=3, delay=1.0)(mock_func)
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await decorated()
    assert res == "Success"
    assert mock_func.call_count == 2
    mock_sleep.assert_called_once_with(1.0)

@pytest.mark.asyncio
async def test_retry_on_network_error_aiohttp_client_error():
    mock_func = AsyncMock()
    mock_func.__name__ = "test_func"
    mock_func.side_effect = aiohttp.ClientError("aiohttp connection lost")

    decorated = retry_on_network_error(retries=3, delay=1.0)(mock_func)
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(aiohttp.ClientError):
            await decorated()
        assert mock_func.call_count == 1
        mock_sleep.assert_not_called()

@pytest.mark.asyncio
async def test_binance_connector_mock():
    from futures_portfolio.connector import BinanceConnectorMock
    mock_conn = BinanceConnectorMock()

    assert await mock_conn.get_exchange_info() == {"symbols": []}
    assert await mock_conn.get_hedge_mode() is True
    assert await mock_conn.set_leverage("BTCUSDT", 10) is None
    assert await mock_conn.set_margin_type("BTCUSDT", "CROSS") is None
    assert await mock_conn.get_mark_prices(["BTCUSDT", "ETHUSDT"]) == {"BTCUSDT": 60000.0, "ETHUSDT": 60000.0}
    assert await mock_conn.get_positions() == {}
    assert await mock_conn.get_margin_ratio() == {"margin_ratio": 10.0}
    assert await mock_conn.verify_connection() is None
