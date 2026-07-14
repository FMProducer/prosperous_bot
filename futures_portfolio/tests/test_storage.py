import pytest
import os
import json
import asyncio
from unittest.mock import patch, mock_open, AsyncMock
from futures_portfolio.core.storage import safe_load_json, safe_load_json_sync, safe_save_json, safe_save_json_sync

def test_safe_load_json_sync_file_exists():
    data = {"key": "value"}
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(data))):
        res = safe_load_json_sync("test.json", {})
        assert res == data

def test_safe_load_json_sync_file_missing():
    with patch("os.path.exists", return_value=False):
        res = safe_load_json_sync("missing.json", {"default": True})
        assert res == {"default": True}

def test_safe_load_json_sync_empty_file():
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="")):
        res = safe_load_json_sync("empty.json", {"default": True})
        assert res == {"default": True}

def test_safe_load_json_sync_corrupted_json():
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="{invalid}")):
        # Should retry 5 times then return default
        res = safe_load_json_sync("corrupted.json", {"default": True}, retries=2)
        assert res == {"default": True}

@pytest.mark.asyncio
async def test_safe_load_json_async():
    data = {"key": "value"}
    with patch("futures_portfolio.core.storage.safe_load_json_sync", return_value=data):
        res = await safe_load_json("test.json", {})
        assert res == data

def test_safe_save_json_sync_atomic(tmp_path):
    path = tmp_path / "test.json"
    data = {"key": "value"}
    safe_save_json_sync(str(path), data)
    assert path.exists()
    with open(path, "r") as f:
        assert json.load(f) == data

def test_safe_save_json_sync_permission_error():
    with patch("os.replace", side_effect=PermissionError), \
         patch("time.sleep", return_value=None), \
         patch("tempfile.mkstemp", return_value=(0, "temp_path")):
        # Mock fdopen to return a context manager
        with patch("os.fdopen", mock_open()), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            with pytest.raises(PermissionError):
                safe_save_json_sync("test.json", {})

@pytest.mark.asyncio
async def test_safe_save_json_async():
    with patch("futures_portfolio.core.storage.safe_save_json_sync") as mock_sync:
        await safe_save_json("test.json", {"data": 1})
        mock_sync.assert_called_once_with("test.json", {"data": 1})

@pytest.mark.asyncio
async def test_safe_load_json_retries_and_timeout():
    with patch("futures_portfolio.core.storage.safe_load_json_sync", side_effect=Exception("Failed")), \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await safe_load_json("test.json", {"default": True}, retries=3)
        assert res == {"default": True}
        assert mock_sleep.call_count == 2

def test_safe_save_json_sync_remove_error():
    # Covers lines 73-74
    with patch("tempfile.mkstemp", return_value=(0, "temp_path")), \
         patch("os.fdopen", mock_open()), \
         patch("os.replace", side_effect=[Exception("Atomic failed")]), \
         patch("os.path.exists", return_value=True), \
         patch("os.remove", side_effect=OSError("Remove failed")):
        try:
            safe_save_json_sync("test.json", {})
        except Exception as e:
            if str(e) != "Atomic failed":
                raise
