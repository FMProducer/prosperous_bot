import pytest
import json
import os
import asyncio
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
async def test_worker_loop():
    # Mocking Path.mkdir
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
