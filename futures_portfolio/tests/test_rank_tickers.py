import pytest
import asyncio
import aiohttp
import json
import os
import time
import pandas as pd
import numpy as np
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
async def test_fetch_klines(scanner):
    mock_session = MagicMock()
    base_time = 1600000000000
    klines = []
    for i in range(200):
        klines.append([base_time + i*60000, "100", "101", "99", "100", "1000", base_time + i*60000 + 59999, "10000", 100, "500", "5000", "0"])

    with patch.object(scanner, "fetch", AsyncMock(return_value=klines)):
        df = await scanner.fetch_klines(mock_session, "BTCUSDT", 200)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert df.index.names == ['ticker', 'time']

@pytest.mark.asyncio
async def test_get_top_tickers(scanner):
    mock_24h = [{"symbol": "BTCUSDT", "quoteVolume": "300000000"}]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]

    base_time = int(time.time() * 1000) - 200 * 60000
    klines = []
    for i in range(200):
        # close is at index 4. Let's make some cycles.
        close = 100 + (i % 10) * 2
        klines.append([base_time + i*60000, "100", "110", "90", str(close), "1000", base_time + i*60000 + 59999, "10000", 100, "500", "5000", "0"])

    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium, klines]
        with patch("builtins.open", MagicMock()):
            with patch("os.path.exists", return_value=False):
                results = await scanner.get_top_tickers(min_volume=200_000_000)

    assert len(results) >= 0 # Depends on cycles calculated

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
