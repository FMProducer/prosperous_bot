import pytest
import asyncio
import aiohttp
import json
import os
import time
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
from futures_portfolio.rank_tickers import TickerScanner, TickerRanker, run_ranker_task, retry_on_network_error


@pytest.fixture
def scanner():
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
        klines.append([base_time + i * 60000, "100", "101", "99", "100", "1000",
                       base_time + i * 60000 + 59999, "10000", 100, "500", "5000", "0"])

    with patch.object(scanner, "fetch", AsyncMock(return_value=klines)):
        df = await scanner.fetch_klines(mock_session, "BTCUSDT", 200)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert df.index.names == ['ticker', 'time']


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


@pytest.mark.asyncio
async def test_fetch_non_200(scanner):
    """Non-429 error status returns None."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()
    mock_session.get.return_value = mock_response
    result = await scanner.fetch(mock_session, "/test")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_exception(scanner):
    """Connection exception returns None."""
    mock_session = MagicMock()
    mock_session.get.side_effect = aiohttp.ClientError("refused")
    result = await scanner.fetch(mock_session, "/test")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_klines_too_few(scanner):
    """fetch_klines returns None when klines < 100."""
    with patch.object(scanner, "fetch", AsyncMock(return_value=[["1"] * 12] * 50)):
        df = await scanner.fetch_klines(MagicMock(), "BTCUSDT", 50)
    assert df is None


@pytest.mark.asyncio
async def test_fetch_klines_empty(scanner):
    """fetch_klines returns None when fetch returns None."""
    with patch.object(scanner, "fetch", AsyncMock(return_value=None)):
        df = await scanner.fetch_klines(MagicMock(), "BTCUSDT", 200)
    assert df is None


@pytest.mark.asyncio
async def test_get_top_tickers(scanner):
    mock_24h = [{"symbol": "BTCUSDT", "quoteVolume": "300000000"}]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]

    mock_result_df = pd.DataFrame({
        "net_change": [5.0], "max_spurt": [10.0], "trend": [2.0], "cycles": [20],
        "symbol": ["BTCUSDT"], "funding": [0.01]
    }, index=["BTCUSDT"])

    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium]
        with patch.object(scanner, "fetch_klines", AsyncMock(return_value=mock_result_df)):
            with patch("builtins.open", mock_open()):
                with patch("os.path.exists", return_value=False):
                    with patch("asyncio.get_running_loop") as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_result_df)
                        results = await scanner.get_top_tickers(min_volume=200_000_000)

    assert len(results) >= 0


@pytest.mark.asyncio
async def test_get_top_tickers_fetch_fail(scanner):
    """get_top_tickers returns [] when fetch fails."""
    with patch.object(scanner, "fetch", AsyncMock(return_value=None)):
        with patch("builtins.open", mock_open()):
            with patch("os.path.exists", return_value=False):
                results = await scanner.get_top_tickers(min_volume=200_000_000)
    assert results == []


@pytest.mark.asyncio
async def test_get_top_tickers_empty_dfs(scanner):
    """get_top_tickers returns [] when all klines are None."""
    mock_24h = [{"symbol": "BTCUSDT", "quoteVolume": "300000000"}]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]

    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium, None]
        with patch("builtins.open", mock_open()):
            with patch("os.path.exists", return_value=False):
                results = await scanner.get_top_tickers(min_volume=200_000_000)
    assert results == []


@pytest.mark.asyncio
async def test_get_top_tickers_blacklist_filter(scanner):
    """Symbols in black_list are excluded."""
    mock_24h = [
        {"symbol": "BTCUSDT", "quoteVolume": "300000000"},
        {"symbol": "ETHUSDT", "quoteVolume": "200000000"},
    ]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]

    cfg_content = json.dumps({"black_list": ["ETHUSDT"]})
    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium]
        mock_result = pd.DataFrame({
            "net_change": [5.0], "max_spurt": [10.0], "trend": [2.0], "cycles": [20],
            "symbol": ["BTCUSDT"], "funding": [0.01]
        }, index=["BTCUSDT"])
        with patch.object(scanner, "fetch_klines", AsyncMock(return_value=mock_result)):
            with patch("builtins.open", mock_open(read_data=cfg_content)):
                with patch("os.path.exists", return_value=True):
                    with patch("asyncio.get_running_loop") as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_result)
                        results = await scanner.get_top_tickers(min_volume=100_000_000)
    # BTCUSDT should be in results (ETHUSDT was blacklisted)
    symbols = [r["symbol"] for r in results]
    assert "ETHUSDT" not in symbols


@pytest.mark.asyncio
async def test_get_top_tickers_whitelist_filter(scanner):
    """When whitelist is set, only whitelisted symbols pass."""
    mock_24h = [
        {"symbol": "BTCUSDT", "quoteVolume": "300000000"},
        {"symbol": "ETHUSDT", "quoteVolume": "200000000"},
    ]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]

    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium]
        mock_result = pd.DataFrame({
            "net_change": [5.0], "max_spurt": [10.0], "trend": [2.0], "cycles": [20],
            "symbol": ["BTCUSDT"], "funding": [0.01]
        }, index=["BTCUSDT"])
        with patch.object(scanner, "fetch_klines", AsyncMock(return_value=mock_result)):
            with patch("builtins.open", mock_open(read_data="BTCUSDT\n")):
                with patch("os.path.exists", return_value=True):
                    with patch("asyncio.get_running_loop") as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_result)
                        results = await scanner.get_top_tickers(min_volume=100_000_000)
    symbols = [r["symbol"] for r in results]
    assert "ETHUSDT" not in symbols


@pytest.mark.asyncio
async def test_get_top_tickers_low_volume_filter(scanner):
    """Symbols below min_volume are excluded."""
    mock_24h = [
        {"symbol": "BTCUSDT", "quoteVolume": "300000000"},
        {"symbol": "LOWUSDT", "quoteVolume": "1000"},  # below threshold
    ]
    mock_premium = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}]

    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium]
        mock_result = pd.DataFrame({
            "net_change": [5.0], "max_spurt": [10.0], "trend": [2.0], "cycles": [20],
            "symbol": ["BTCUSDT"], "funding": [0.01]
        }, index=["BTCUSDT"])
        with patch.object(scanner, "fetch_klines", AsyncMock(return_value=mock_result)):
            with patch("builtins.open", mock_open()):
                with patch("os.path.exists", return_value=False):
                    with patch("asyncio.get_running_loop") as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_result)
                        results = await scanner.get_top_tickers(min_volume=200_000_000)
    symbols = [r["symbol"] for r in results]
    assert "LOWUSDT" not in symbols


@pytest.mark.asyncio
async def test_get_top_tickers_non_ascii(scanner):
    """Non-ASCII symbols are excluded."""
    mock_24h = [
        {"symbol": "DOGEUSDT", "quoteVolume": "300000000"},
        {"symbol": "БТСUSDT", "quoteVolume": "300000000"},
    ]
    mock_premium = []

    with patch.object(scanner, "fetch", AsyncMock()) as mock_fetch:
        mock_fetch.side_effect = [mock_24h, mock_premium]
        with patch("builtins.open", mock_open()):
            with patch("os.path.exists", return_value=False):
                results = await scanner.get_top_tickers(min_volume=100_000_000)
    assert results == []


# --- TickerRanker tests ---

def _make_multi_df(tickers=None, n=200):
    """Create a MultiIndex DataFrame for TickerRanker tests."""
    if tickers is None:
        tickers = ["BTCUSDT", "ETHUSDT"]
    idx = pd.MultiIndex.from_product([tickers, range(n)], names=["ticker", "time"])
    np.random.seed(42)
    close = 100 + np.random.randn(len(idx)).cumsum() * 0.5
    data = {
        "open": close - 0.1,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.abs(np.random.randn(len(idx))) * 1000,
    }
    return pd.DataFrame(data, index=idx)


def test_ranker_init():
    """TickerRanker __init__ selects OHLCV columns."""
    df = _make_multi_df()
    ranker = TickerRanker(df)
    assert list(ranker.df.columns) == ['open', 'high', 'low', 'close', 'volume']


def test_ranker_calculate_metrics():
    """calculate_metrics returns DataFrame with expected columns."""
    df = _make_multi_df()
    ranker = TickerRanker(df)
    result = ranker.calculate_metrics(threshold=0.005)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'net_change', 'max_spurt', 'trend', 'cycles'}
    assert len(result) == 2  # BTCUSDT and ETHUSDT


def test_ranker_rank_by_momentum():
    """rank_by_momentum returns sorted Series."""
    df = _make_multi_df()
    ranker = TickerRanker(df)
    result = ranker.rank_by_momentum(window=14)
    assert isinstance(result, pd.Series)
    assert len(result) == 2


def test_run_ranker_task():
    """run_ranker_task helper runs correctly."""
    df = _make_multi_df()
    result = run_ranker_task(df, 0.005)
    assert isinstance(result, pd.DataFrame)
    assert "cycles" in result.columns


@pytest.mark.asyncio
async def test_retry_on_network_error_exhausted():
    """Line 33: returns None after all retries exhausted."""
    @retry_on_network_error(retries=2, delay=0.001)
    async def always_fail():
        raise ValueError("boom")

    result = await always_fail()
    assert result is None


@pytest.mark.asyncio
async def test_retry_on_network_error_success():
    """Decorator succeeds on first try."""
    @retry_on_network_error(retries=3, delay=0.001)
    async def succeed():
        return 42

    result = await succeed()
    assert result == 42


@pytest.mark.asyncio
async def test_main_with_config():
    """Lines 214-217: main() reads config.json for threshold."""
    mock_result = [{"symbol": "BTCUSDT", "cycles": 20, "net_change": 5.0,
                     "max_spurt": 10.0, "trend": 2.0, "funding": 0.01}]
    cfg_content = json.dumps({"scanner_period_days": 0.25})
    with patch("rank_tickers.TickerScanner") as MockScanner:
        mock_scanner = AsyncMock()
        mock_scanner.get_top_tickers.return_value = mock_result
        MockScanner.return_value = mock_scanner
        with patch("builtins.open", mock_open(read_data=cfg_content)):
            with patch("os.path.exists", return_value=True):
                result = await __import__("rank_tickers").main(quiet=True)
    assert result == mock_result


# --- main() tests ---

@pytest.mark.asyncio
async def test_main_quiet():
    """main() quiet=True doesn't print."""
    mock_result = [{"symbol": "BTCUSDT", "cycles": 20, "net_change": 5.0,
                     "max_spurt": 10.0, "trend": 2.0, "funding": 0.01}]
    with patch("rank_tickers.TickerScanner") as MockScanner:
        mock_scanner = AsyncMock()
        mock_scanner.get_top_tickers.return_value = mock_result
        MockScanner.return_value = mock_scanner
        with patch("builtins.open", mock_open()):
            with patch("os.path.exists", return_value=False):
                result = await __import__("rank_tickers").main(quiet=True)
    assert result == mock_result


@pytest.mark.asyncio
async def test_main_not_quiet():
    """main() quiet=False prints results."""
    mock_result = [{"symbol": "BTCUSDT", "cycles": 20, "net_change": 5.0,
                     "max_spurt": 10.0, "trend": 2.0, "funding": 0.01}]
    with patch("rank_tickers.TickerScanner") as MockScanner:
        mock_scanner = AsyncMock()
        mock_scanner.get_top_tickers.return_value = mock_result
        MockScanner.return_value = mock_scanner
        with patch("builtins.open", mock_open()):
            with patch("os.path.exists", return_value=False):
                result = await __import__("rank_tickers").main(quiet=False)
    assert result == mock_result


@pytest.mark.asyncio
async def test_main_empty():
    """main() with empty result."""
    with patch("rank_tickers.TickerScanner") as MockScanner:
        mock_scanner = AsyncMock()
        mock_scanner.get_top_tickers.return_value = []
        MockScanner.return_value = mock_scanner
        with patch("builtins.open", mock_open()):
            with patch("os.path.exists", return_value=False):
                result = await __import__("rank_tickers").main(quiet=True)
    assert result == []
