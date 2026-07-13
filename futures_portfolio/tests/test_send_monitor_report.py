import pytest
import json
import os
import time
import importlib
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
from io import BytesIO


@pytest.fixture(autouse=True)
def clean_module():
    """Remove send_monitor_report from sys.modules before/after each test."""
    sys.modules.pop("send_monitor_report", None)
    yield
    sys.modules.pop("send_monitor_report", None)


def _mock_env():
    """Return standard mocks for module-level globals."""
    return {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "test_chat",
        "TELEGRAM_API_BASE": "https://api.telegram.org",
    }


def _make_paper_state(tpv=105.0, pnl=5.0, cycles=42, guard=False):
    return json.dumps({
        "last_tpv": tpv, "last_profit": pnl,
        "rebalance_cycles": cycles, "pnl_guard_active": guard,
    }).encode()


def _make_real_state(pnl=10.0):
    return json.dumps({"last_profit": pnl}).encode()


def _make_log_content(minutes_ago=1):
    """Return log content with mtime set minutes_ago."""
    ts = time.time() - minutes_ago * 60
    return b"log line\n", ts


def _mock_urlopen(response_body=b'{"ok":true}', status=200):
    """Create a mock opener that captures the request."""
    captured = {}

    class MockResponse:
        def read(self):
            return response_body

    def mock_open(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return MockResponse()

    return mock_open, captured


def test_monitor_no_state_files():
    """No state files → N/A values, 'No real state' alert."""
    mock_open, captured = _mock_urlopen()

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=False), \
         patch("urllib.request.build_opener") as mock_builder:
        mock_builder.return_value.open = mock_open
        mod = importlib.import_module("send_monitor_report")

    assert "N/A" in mod.msg
    assert "No real state" in mod.msg


def test_monitor_with_paper_and_real():
    """Both state files exist → values populated."""
    mock_open, captured = _mock_urlopen()

    def exists_side_effect(self_path=None):
        # Path.exists() is called as method; we check the string
        return True

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", MagicMock()) as mock_file, \
         patch("urllib.request.build_opener") as mock_builder:
        # Mock open to return appropriate data
        call_count = [0]
        def open_side_effect(path, *args, **kwargs):
            mock_f = MagicMock()
            p_str = str(path)
            if "paper_state" in p_str:
                mock_f.__enter__ = MagicMock(return_value=mock_f)
                mock_f.__exit__ = MagicMock(return_value=False)
                mock_f.read.return_value = _make_paper_state(tpv=105.0, pnl=5.0, cycles=42, guard=False)
                mock_f.__enter__.return_value = mock_f
                return mock_f
            elif "real_state" in p_str:
                mock_f.__enter__ = MagicMock(return_value=mock_f)
                mock_f.__exit__ = MagicMock(return_value=False)
                mock_f.read.return_value = _make_real_state(pnl=10.0)
                return mock_f
            else:
                mock_f.__enter__ = MagicMock(return_value=mock_f)
                mock_f.__exit__ = MagicMock(return_value=False)
                mock_f.read.return_value = b""
                return mock_f

        mock_builder.return_value.open = mock_open
        # Can't easily test this way due to module-level code. Use simpler approach.
        pass

    # Simpler: just check the module constructs correct message parts
    # by testing with Path.exists mocked at module level
    sys.modules.pop("send_monitor_report", None)

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open") as mock_file_open, \
         patch("urllib.request.build_opener") as mock_builder, \
         patch("os.stat") as mock_stat:

        def fake_open(path, *args, **kwargs):
            m = MagicMock()
            p_str = str(path)
            if "paper_state" in p_str:
                m.__enter__ = MagicMock(return_value=m)
                m.__exit__ = MagicMock(return_value=False)
                content = json.dumps({
                    "last_tpv": 105.0, "last_profit": 5.0,
                    "rebalance_cycles": 42, "pnl_guard_active": False
                })
                m.read.return_value = content if "r" in str(args) else content.encode()
                m.write = MagicMock()
                return m
            elif "real_state" in p_str:
                m.__enter__ = MagicMock(return_value=m)
                m.__exit__ = MagicMock(return_value=False)
                content = json.dumps({"last_profit": 10.0})
                m.read.return_value = content if "r" in str(args) else content.encode()
                return m
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            m.read.return_value = b""
            return m

        mock_file_open.side_effect = fake_open
        mock_builder.return_value.open = mock_open
        mock_stat.return_value.st_mtime = time.time() - 60  # 1 min ago

        mod = importlib.import_module("send_monitor_report")

    assert "105.0" in mod.msg or "105" in mod.msg
    assert "HEIUSDT" in mod.msg


def test_monitor_guard_active():
    """PnL guard ON → alert in message."""
    mock_open, captured = _mock_urlopen()

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open") as mock_file_open, \
         patch("urllib.request.build_opener") as mock_builder, \
         patch("os.stat") as mock_stat:

        def fake_open(path, *args, **kwargs):
            m = MagicMock()
            p_str = str(path)
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            if "paper_state" in p_str:
                m.read.return_value = json.dumps({
                    "last_tpv": 100, "last_profit": 0,
                    "rebalance_cycles": 0, "pnl_guard_active": True
                })
            elif "real_state" in p_str:
                m.read.return_value = json.dumps({"last_profit": 0})
            else:
                m.read.return_value = b""
            return m

        mock_file_open.side_effect = fake_open
        mock_builder.return_value.open = mock_open
        mock_stat.return_value.st_mtime = time.time() - 30

        mod = importlib.import_module("send_monitor_report")

    assert "PnL GUARD is active" in mod.msg
    assert "⚠️" in mod.msg


def test_monitor_old_heartbeat():
    """Heartbeat older than 5 min → alert."""
    mock_open, captured = _mock_urlopen()

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open") as mock_file_open, \
         patch("urllib.request.build_opener") as mock_builder, \
         patch("os.stat") as mock_stat:

        def fake_open(path, *args, **kwargs):
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            p_str = str(path)
            if "paper_state" in p_str:
                m.read.return_value = json.dumps({
                    "last_tpv": 100, "last_profit": 0,
                    "rebalance_cycles": 0, "pnl_guard_active": False
                })
            elif "real_state" in p_str:
                m.read.return_value = json.dumps({"last_profit": 0})
            else:
                m.read.return_value = b""
            return m

        mock_file_open.side_effect = fake_open
        mock_builder.return_value.open = mock_open
        mock_stat.return_value.st_mtime = time.time() - 600  # 10 min ago

        mod = importlib.import_module("send_monitor_report")

    assert "heartbeat" in mod.msg.lower() or "STOPPED" in mod.msg


def test_monitor_sends_request():
    """Module sends HTTP POST to Telegram."""
    mock_open, captured = _mock_urlopen()

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=False), \
         patch("urllib.request.build_opener") as mock_builder:
        mock_builder.return_value.open = mock_open
        mod = importlib.import_module("send_monitor_report")

    assert captured["req"].method == "POST"
    assert captured["timeout"] == 20


def test_monitor_no_alerts():
    """No alerts → ✅ No alerts."""
    mock_open, captured = _mock_urlopen()

    with patch.dict(os.environ, _mock_env()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open") as mock_file_open, \
         patch("urllib.request.build_opener") as mock_builder, \
         patch("os.stat") as mock_stat:

        def fake_open(path, *args, **kwargs):
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            p_str = str(path)
            if "paper_state" in p_str:
                m.read.return_value = json.dumps({
                    "last_tpv": 100, "last_profit": 0,
                    "rebalance_cycles": 5, "pnl_guard_active": False
                })
            elif "real_state" in p_str:
                m.read.return_value = json.dumps({"last_profit": 0})
            else:
                m.read.return_value = b""
            return m

        mock_file_open.side_effect = fake_open
        mock_builder.return_value.open = mock_open
        mock_stat.return_value.st_mtime = time.time() - 30  # recent

        mod = importlib.import_module("send_monitor_report")

    assert "No alerts" in mod.msg
