import pytest
import os
import json
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from futures_portfolio.core.notifier import TelegramNotifier

@pytest.fixture
def env_setup():
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "test_chat_id",
        "TELEGRAM_API_BASE": "https://api.telegram.org",
        "TG_PROXY": "socks5://127.0.0.1:10808"
    }):
        yield

def test_notifier_init(env_setup):
    notifier = TelegramNotifier()
    assert notifier.enabled is True

def test_notifier_disabled():
    with patch.dict(os.environ, {}, clear=True):
        notifier = TelegramNotifier()
        assert notifier.enabled is False

def test_load_config_exception():
    notifier = TelegramNotifier()
    with patch("builtins.open", side_effect=Exception("Read error")):
        config = notifier._load_config()
        assert config == {}

@pytest.mark.asyncio
async def test_session_creation(env_setup):
    notifier = TelegramNotifier()
    # Mock ProxyConnector and ClientSession
    mock_connector = MagicMock()
    with patch("futures_portfolio.core.notifier.ProxyConnector.from_url", return_value=mock_connector) as mock_from_url:
        with patch("futures_portfolio.core.notifier.aiohttp.ClientSession") as mock_session_cls:
            session = await notifier._get_session()
            mock_from_url.assert_called_with("socks5://127.0.0.1:10808", ssl=False)
            mock_session_cls.assert_called_once_with(connector=mock_connector, trust_env=False)
            await notifier.close()

@pytest.mark.asyncio
async def test_queue_message_text(env_setup, tmp_path):
    notifier = TelegramNotifier()
    # Force use_queue
    notifier.use_queue = True
    notifier.queue_dir = str(tmp_path / "signals" / "telegram_queue")
    os.makedirs(notifier.queue_dir, exist_ok=True)

    result = await notifier.send_message("Test Queue", force_direct=False)
    assert result is True

    # Check that a file was created in queue_dir
    files = os.listdir(notifier.queue_dir)
    assert len(files) == 1
    with open(os.path.join(notifier.queue_dir, files[0]), "r") as f:
        data = json.load(f)
        assert data["type"] == "text"
        assert data["text"] == "Test Queue"

@pytest.mark.asyncio
async def test_queue_message_exception(env_setup):
    notifier = TelegramNotifier()
    notifier.use_queue = True
    # Let's cause an exception by making queue_dir invalid or patching json.dump
    with patch("builtins.open", side_effect=Exception("Write error")):
        result = await notifier._queue_message({"some": "data"})
        assert result is False

@pytest.mark.asyncio
async def test_send_message_direct_success(env_setup):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_resp
    notifier._session = mock_session

    result = await notifier.send_message("Test Direct", force_direct=True)
    assert result is True

@pytest.mark.asyncio
async def test_send_message_direct_429_short_retry(env_setup):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    # First attempt: 429 with short retry
    mock_resp_1 = MagicMock()
    mock_resp_1.status = 429
    mock_resp_1.text = AsyncMock(return_value='{"parameters": {"retry_after": 2}}')
    mock_resp_1.__aenter__ = AsyncMock(return_value=mock_resp_1)
    mock_resp_1.__aexit__ = AsyncMock(return_value=None)

    # Second attempt: 200 success
    mock_resp_2 = MagicMock()
    mock_resp_2.status = 200
    mock_resp_2.__aenter__ = AsyncMock(return_value=mock_resp_2)
    mock_resp_2.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.side_effect = [mock_resp_1, mock_resp_2]
    notifier._session = mock_session

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await notifier.send_message("Test Direct 429", force_direct=True)
        assert result is True
        mock_sleep.assert_called_with(2.5)

@pytest.mark.asyncio
async def test_send_message_direct_429_long_retry_skip(env_setup):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    # 429 with too long retry
    mock_resp = MagicMock()
    mock_resp.status = 429
    mock_resp.text = AsyncMock(return_value='{"parameters": {"retry_after": 100}}')
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_resp
    notifier._session = mock_session

    result = await notifier.send_message("Test Direct 429 Long", force_direct=True)
    assert result is False

@pytest.mark.asyncio
async def test_send_message_direct_api_error(env_setup):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    mock_resp = MagicMock()
    mock_resp.status = 400
    mock_resp.text = AsyncMock(return_value='{"description": "Bad Request"}')
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_resp
    notifier._session = mock_session

    result = await notifier.send_message("Test Direct Error", force_direct=True)
    assert result is False

@pytest.mark.asyncio
async def test_send_message_connection_error(env_setup):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.side_effect = Exception("Network Disconnected")
    notifier._session = mock_session

    with patch("asyncio.sleep", AsyncMock()):
        result = await notifier.send_message("Test Connection Error", force_direct=True, max_retries=2)
        assert result is False

@pytest.mark.asyncio
async def test_send_photo_missing_path(env_setup):
    notifier = TelegramNotifier()
    result = await notifier.send_photo("/non/existent/path.png")
    assert result is False

@pytest.mark.asyncio
async def test_send_photo_queue(env_setup, tmp_path):
    notifier = TelegramNotifier()
    notifier.use_queue = True
    notifier.queue_dir = str(tmp_path / "signals" / "telegram_queue")
    os.makedirs(notifier.queue_dir, exist_ok=True)

    dummy_file = tmp_path / "test.png"
    dummy_file.write_text("dummy image content")

    result = await notifier.send_photo(str(dummy_file), "My Caption", force_direct=False)
    assert result is True

    files = os.listdir(notifier.queue_dir)
    assert len(files) == 1
    with open(os.path.join(notifier.queue_dir, files[0]), "r") as f:
        data = json.load(f)
        assert data["type"] == "photo"
        assert data["path"] == str(dummy_file)
        assert data["caption"] == "My Caption"

@pytest.mark.asyncio
async def test_send_photo_direct_success(env_setup, tmp_path):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    dummy_file = tmp_path / "test.png"
    dummy_file.write_text("dummy image content")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_resp
    notifier._session = mock_session

    result = await notifier.send_photo(str(dummy_file), "Direct photo", force_direct=True)
    assert result is True

@pytest.mark.asyncio
async def test_send_photo_direct_429(env_setup, tmp_path):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    dummy_file = tmp_path / "test.png"
    dummy_file.write_text("dummy image content")

    mock_resp_1 = MagicMock()
    mock_resp_1.status = 429
    mock_resp_1.text = AsyncMock(return_value='{"parameters": {"retry_after": 3}}')
    mock_resp_1.__aenter__ = AsyncMock(return_value=mock_resp_1)
    mock_resp_1.__aexit__ = AsyncMock(return_value=None)

    mock_resp_2 = MagicMock()
    mock_resp_2.status = 200
    mock_resp_2.__aenter__ = AsyncMock(return_value=mock_resp_2)
    mock_resp_2.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.side_effect = [mock_resp_1, mock_resp_2]
    notifier._session = mock_session

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await notifier.send_photo(str(dummy_file), "Direct photo 429", force_direct=True)
        assert result is True
        mock_sleep.assert_called_with(3.5)

@pytest.mark.asyncio
async def test_send_photo_direct_429_long(env_setup, tmp_path):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    dummy_file = tmp_path / "test.png"
    dummy_file.write_text("dummy image content")

    mock_resp_1 = MagicMock()
    mock_resp_1.status = 429
    mock_resp_1.text = AsyncMock(return_value='{"parameters": {"retry_after": 50}}')
    mock_resp_1.__aenter__ = AsyncMock(return_value=mock_resp_1)
    mock_resp_1.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_resp_1
    notifier._session = mock_session

    result = await notifier.send_photo(str(dummy_file), "Direct photo 429 Long", force_direct=True)
    assert result is False

@pytest.mark.asyncio
async def test_send_photo_direct_api_error(env_setup, tmp_path):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    dummy_file = tmp_path / "test.png"
    dummy_file.write_text("dummy")

    mock_resp = MagicMock()
    mock_resp.status = 400
    mock_resp.text = AsyncMock(return_value='{"description": "Bad Request"}')
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.return_value = mock_resp
    notifier._session = mock_session

    result = await notifier.send_photo(str(dummy_file), "Error photo", force_direct=True)
    assert result is False

@pytest.mark.asyncio
async def test_send_photo_exception(env_setup, tmp_path):
    notifier = TelegramNotifier()
    notifier.use_queue = False

    dummy_file = tmp_path / "test.png"
    dummy_file.write_text("dummy")

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post.side_effect = Exception("photo exception")
    notifier._session = mock_session

    with patch("asyncio.sleep", AsyncMock()):
        result = await notifier.send_photo(str(dummy_file), "Ex photo", force_direct=True, max_retries=2)
        assert result is False

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
        await notifier.send_status("BTCUSDT", 10000.0, 500.0, 10, 10500.0, total_balance=12000.0, bnb_balance=0.5)
        mock_send.assert_called_once()
