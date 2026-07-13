import pytest
import json
import os
import asyncio
import aiohttp
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
import telegram_sender

@pytest.mark.asyncio
async def test_send_to_telegram_success():
    session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    # Use MagicMock for the context manager returned by session.post
    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.return_value = mock_response
    session.post.return_value = mock_post_cm
    
    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"):
        res = await telegram_sender.send_to_telegram(session, {"text": "hello"})
        assert res is True

@pytest.mark.asyncio
async def test_send_to_telegram_rate_limit():
    session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.text.return_value = json.dumps({"parameters": {"retry_after": 1}})
    
    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.return_value = mock_response
    session.post.return_value = mock_post_cm
    
    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"), \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await telegram_sender.send_to_telegram(session, {"text": "hello"})
        assert res is False
        mock_sleep.assert_called_with(1)

@pytest.mark.asyncio
async def test_send_to_telegram_rate_limit_invalid_json():
    """Line 44-45: bare except when rate-limit response is not valid JSON."""
    session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.text.return_value = "not json"

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.return_value = mock_response
    session.post.return_value = mock_post_cm

    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"), \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await telegram_sender.send_to_telegram(session, {"text": "hello"})
        assert res is False
        mock_sleep.assert_called_with(30)  # fallback retry_after


@pytest.mark.asyncio
async def test_send_to_telegram_timeout():
    """Line 52-54: asyncio.TimeoutError path."""
    session = MagicMock()
    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.side_effect = asyncio.TimeoutError()
    session.post.return_value = mock_post_cm

    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"):
        res = await telegram_sender.send_to_telegram(session, {"text": "hello"})
        assert res is False


@pytest.mark.asyncio
async def test_send_to_telegram_connection_error():
    """Line 55-57: generic exception path."""
    session = MagicMock()
    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.side_effect = aiohttp.ClientError("connection refused")
    session.post.return_value = mock_post_cm

    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"):
        res = await telegram_sender.send_to_telegram(session, {"text": "hello"})
        assert res is False


@pytest.mark.asyncio
async def test_worker_missing_credentials():
    """Line 62-63: worker returns early when TOKEN or CHAT_ID is missing."""
    with patch("telegram_sender.TOKEN", None), \
         patch("telegram_sender.CHAT_ID", "test_chat"):
        await telegram_sender.worker()
    # Should return without doing anything


@pytest.mark.asyncio
async def test_worker_empty_queue():
    """Line 74-75: worker sleeps when queue is empty."""
    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"), \
         patch("pathlib.Path.mkdir"), \
         patch("telegram_sender.ProxyConnector.from_url"), \
         patch("aiohttp.ClientSession") as mock_session_cls, \
         patch("pathlib.Path.glob", return_value=[]), \
         patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
        mock_session = MagicMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        try:
            await telegram_sender.worker()
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_worker_send_failure():
    """Line 91: worker retries on send failure."""
    with patch("telegram_sender.TOKEN", "test_token"), \
        patch("telegram_sender.CHAT_ID", "test_chat"), \
        patch("pathlib.Path.mkdir"), \
        patch("telegram_sender.ProxyConnector.from_url"), \
        patch("aiohttp.ClientSession") as mock_session_cls, \
        patch("pathlib.Path.glob", return_value=[MagicMock(suffix=".json", name="msg_1.json")]), \
        patch("builtins.open", mock_open(read_data='{"text": "msg"}')), \
        patch("telegram_sender.send_to_telegram", AsyncMock(return_value=False)), \
        patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
       mock_session = MagicMock()
       mock_session_cls.return_value.__aenter__.return_value = mock_session
       try:
           await telegram_sender.worker()
       except asyncio.CancelledError:
           pass


@pytest.mark.asyncio
async def test_send_to_telegram_api_error():
    """Line 50-51: non-429 error response logging."""
    session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text.return_value = "Internal Server Error"

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.return_value = mock_response
    session.post.return_value = mock_post_cm

    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"):
        res = await telegram_sender.send_to_telegram(session, {"text": "hello"})
        assert res is False


@pytest.mark.asyncio
async def test_worker_success():
    """Lines 85-88: worker removes file and sleeps after successful send."""
    with patch("telegram_sender.TOKEN", "test_token"), \
         patch("telegram_sender.CHAT_ID", "test_chat"), \
         patch("pathlib.Path.mkdir"), \
         patch("telegram_sender.ProxyConnector.from_url"), \
         patch("aiohttp.ClientSession") as mock_session_cls, \
         patch("pathlib.Path.glob", return_value=[MagicMock(suffix=".json", name="msg_1.json")]), \
         patch("builtins.open", mock_open(read_data='{"text": "msg"}')), \
         patch("telegram_sender.send_to_telegram", AsyncMock(return_value=True)), \
         patch("os.remove") as mock_remove, \
         patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
        mock_session = MagicMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        try:
            await telegram_sender.worker()
        except asyncio.CancelledError:
            pass
        mock_remove.assert_called()
