import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from futures_portfolio.notifier import TelegramNotifier

@pytest.fixture
def env_setup():
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "test_chat_id",
        "TELEGRAM_API_BASE": "https://api.telegram.org"
    }):
        yield

def test_notifier_init(env_setup):
    notifier = TelegramNotifier()
    assert notifier.enabled is True

def test_notifier_disabled():
    with patch.dict(os.environ, {}, clear=True):
        notifier = TelegramNotifier()
        assert notifier.enabled is False

@pytest.mark.asyncio
async def test_send_message_success(env_setup):
    notifier = TelegramNotifier()
    
    # Мокаем aiohttp.ClientSession.post
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        result = await notifier.send_message("Hello")
        assert result is True

@pytest.mark.asyncio
async def test_send_alert(env_setup):
    notifier = TelegramNotifier()
    with patch.object(notifier, "send_message", AsyncMock()) as mock_send:
        await notifier.send_alert("Title", "Message")
        mock_send.assert_called_once()

@pytest.mark.asyncio
async def test_send_status(env_setup):
    notifier = TelegramNotifier()
    with patch.object(notifier, "send_message", AsyncMock()) as mock_send:
        await notifier.send_status("BTCUSDT", 10000.0, 500.0, 10, 10500.0, 0.5)
        mock_send.assert_called_once()
