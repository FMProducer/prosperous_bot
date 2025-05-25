# c:\Python\Prosperous_Bot\tests\test_exchange.py

import pytest, gate_api
from unittest import mock
import gate_api
from exchange_api import ExchangeAPI
import asyncio
import time
from gate_api import ApiException

@pytest.mark.asyncio
async def test_get_system_time(exch, mocker):
    mocker.patch.object(
        gate_api.SpotApi,
        "get_system_time",
        return_value=gate_api.SystemTime(server_time=123_456_789),
    )
    assert await exch.get_system_time() == 123_456_789

@pytest.mark.asyncio
async def test_get_current_price(exch, mocker):
    mocker.patch.object(
        gate_api.SpotApi,
        "list_tickers",
        return_value=[gate_api.Ticker(currency_pair="BTC_USDT", last="42000")]
    )
    assert await exch.get_current_price("BTC_USDT") == 42_000.0

@pytest.mark.asyncio
async def test_create_futures_order(exch, mocker):
    # мок: FuturesApi.create_futures_order должен просто вернуть echo-dict
    mock_create = mocker.patch.object(
        gate_api.FuturesApi,
        "create_futures_order",
        return_value={"status": "open", "size": "1"},
    )
    res = await exch.create_futures_order(
        contract="BTC_USDT",
        side="long",
        qty=1,
        reduce_only=False,
    )
    assert res["status"] == "open"
    # убедимся, что вызвали API ровно один раз
    mock_create.assert_called_once()

# ---- 4. create_spot_order ----
@pytest.mark.asyncio
async def test_create_spot_order(exch, mocker):
    mocker.patch.object(
        gate_api.SpotApi,
        "create_order",
        return_value={"id": "123", "side": "buy"},
    )
    res = await exch.create_spot_order(
        pair="BTC_USDT", side="buy", qty=0.01, price=None, post_only=False
    )
    assert res["id"] == "123"

# ---- 5. positions ----
@pytest.mark.asyncio
async def test_positions(exch, mocker):
    mocker.patch.object(
        gate_api.FuturesApi,
        "list_positions",
        return_value=[{"size": "1", "contract": "BTC_USDT"}],
    )
    pos = await exch.positions()
    assert pos[0]["contract"] == "BTC_USDT"

# ---- 6. _safe_call error branch ----
@pytest.mark.asyncio
async def test_safe_call_retries(exch, mocker):
    # заставляем первый вызов кидать 429, второй — успешный
    err = ApiException(status=429, reason="too many requests")
    mock_fn = mocker.Mock(side_effect=[err, "ok"])
    out = await exch._safe_call(mock_fn)
    assert out == "ok"
    assert mock_fn.call_count == 2

# ---- 7. create_spot_order post_only=True ----
@pytest.mark.asyncio
async def test_create_spot_post_only(exch, mocker):
    mocker.patch.object(
        gate_api.SpotApi,
        "create_order",
        return_value={"id": "999", "iceberg": "0"},
    )
    res = await exch.create_spot_order(
        pair="BTC_USDT", side="sell", qty=0.02, price=50_000, post_only=True
    )
    assert res["id"] == "999"

# ---- 8. _safe_call raises non-retry error ----
@pytest.mark.asyncio
async def test_safe_call_raises(exch, mocker):
    # GateApiException в рантайме имеет .message, добавим его вручную
    err = ApiException(status=400, reason="bad req")
    err.message = "bad req"
    mock_fn = mocker.Mock(side_effect=err)
    with pytest.raises(ApiException):
        await exch._safe_call(mock_fn)

# ---- 9. _sync_time updates offset ----
@pytest.mark.asyncio
async def test_sync_time(exch, mocker):
    # time before call = 0  ➜  заставим функции думать, что прошло 1 ч.
    exch._last_sync = 0
    mocker.patch.object(
        gate_api.SpotApi,
        "get_system_time",
        return_value=gate_api.SystemTime(server_time=2000),
    )
    await exch._sync_time()
    assert exch._time_offset == 2000 - int(exch._last_sync * 1000)
    # ---- 10. _safe_call generic Exception ----
@pytest.mark.asyncio
async def test_safe_call_generic_error(exch, mocker):
    # sync-функция, чтобы исключение возникло внутри to_thread
    def boom():
        raise RuntimeError("net down")
    with pytest.raises(RuntimeError):
        await exch._safe_call(boom)

# ---- 11. create_spot_order limit + post_only ----
@pytest.mark.asyncio
async def test_create_spot_limit_post_only(exch, mocker):
    mocker.patch.object(
        gate_api.SpotApi,
        "create_order",
        return_value={"id": "321", "iceberg": "0"},
    )
    res = await exch.create_spot_order(
        pair="BTC_USDT",
        side="buy",
        qty=0.03,
        price=48_000,    # limit-price → post_only проверка
        post_only=True,
    )
    assert res["id"] == "321"

# ---- 12. _sync_time skip branch ----
@pytest.mark.asyncio
async def test_sync_time_skip(exch, mocker):
    # _last_sync сделаем «только что» → метод должен быстро выйти
    exch._last_sync = time.time()
    called = mocker.patch.object(
        gate_api.SpotApi, "get_system_time", autospec=True
    )
    await exch._sync_time()
    # get_system_time не вызывается
    called.assert_not_called()