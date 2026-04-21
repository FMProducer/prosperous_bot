import pytest
import asyncio
import aiohttp
import json
import os
import time
from unittest.mock import patch, AsyncMock, MagicMock
from futures_portfolio.rank_tickers import TickerScanner

@pytest.fixture
def scanner():
    # Use 2 to avoid deadlock in recursive fetch() call during rate limit test
    return TickerScanner(concurrent_requests=2)

@pytest.mark.asyncio
async def test_fetch_success(scanner):
    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"key": "value"}
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()
    mock_session.get.return_value = mock_response
    result = await scanner.fetch(mock_session, "/test")
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_analyze_ticker_simple(scanner):
    mock_session = MagicMock()
    age_check = [["data"]]
    klines = []
    base_time = 1600000000000
    for i in range(2880):
        klines.append([base_time + i*60000, "100", "101", "99", "100", "1000"])
    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [age_check, klines[:1440], klines[1440:]]
        result = await scanner.analyze_ticker(mock_session, "BTCUSDT", 0.0001)
    assert result is not None

@pytest.mark.asyncio
async def test_get_top_tickers(scanner):
    mock_24h = [{"symbol": "BTCUSDT", "quoteVolume": "300000000"}]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]
    mock_result = {"symbol": "BTCUSDT", "score": 100, "tier": "Tier-2"}
    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium]
        with patch.object(scanner, "analyze_ticker", AsyncMock(return_value=mock_result)):
            with patch("builtins.open", MagicMock()):
                with patch("os.path.exists", return_value=False):
                    results = await scanner.get_top_tickers(min_volume=200_000_000)
    assert len(results) == 1

@pytest.mark.asyncio
async def test_analyze_ticker_full_logic(scanner):
    mock_session = MagicMock()
    age_check = [["data"]]
    klines = []
    base_time = 1600000000000
    for i in range(2880):
        p = 100.0
        if (i // 100) % 2 == 0: p = 102.0
        klines.append([base_time + i*60000, "100", str(p+0.1), str(p-0.1), str(p), "1000"])
    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [age_check, klines[:1440], klines[1440:]]
        res = await scanner.analyze_ticker(mock_session, "TEST", -0.0001)
        assert res["cycles"] >= 8
        assert res["score"] > 0

@pytest.mark.asyncio
async def test_fetch_rate_limit(scanner):
    mock_session = MagicMock()
    mock_response_429 = MagicMock()
    mock_response_429.status = 429
    mock_response_429.headers = {"Retry-After": "0"}
    mock_response_429.__aenter__ = AsyncMock(return_value=mock_response_429)
    mock_response_429.__aexit__ = AsyncMock()
    mock_response_200 = MagicMock()
    mock_response_200.status = 200
    mock_response_200.json = AsyncMock(return_value={"ok": True})
    mock_response_200.__aenter__ = AsyncMock(return_value=mock_response_200)
    mock_response_200.__aexit__ = AsyncMock()
    mock_session.get.side_effect = [mock_response_429, mock_response_200]
    with patch("asyncio.sleep", AsyncMock()):
        res = await scanner.fetch(mock_session, "/api")
        assert res == {"ok": True}
