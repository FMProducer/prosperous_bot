import pytest
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from futures_portfolio import health_check

# Mock CREATE_NO_WINDOW before any health_check calls if not on Windows
if not hasattr(subprocess, "CREATE_NO_WINDOW"):
    subprocess.CREATE_NO_WINDOW = 0

def test_check_pm2_success():
    mock_processes = [
        {"name": "real-btc", "pm2_env": {"status": "online"}},
        {"name": "paper-eth", "pm2_env": {"status": "online"}},
        {"name": "other-process", "pm2_env": {"status": "online"}},
        {"name": "stopped-process", "pm2_env": {"status": "stopped"}},
    ]
    mock_completed_process = MagicMock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = json.dumps(mock_processes)
    mock_completed_process.stderr = ""

    with patch("subprocess.run", return_value=mock_completed_process):
        res = health_check.check_pm2()
        assert res["status"] == "ok"
        assert res["total_online"] == 3
        assert res["real"] == 1
        assert res["paper"] == 1
        assert res["infra"] == 1
        assert res["stopped"] == 1

def test_check_pm2_nonzero_return():
    mock_completed_process = MagicMock()
    mock_completed_process.returncode = 1
    mock_completed_process.stderr = "some pm2 error"

    with patch("subprocess.run", return_value=mock_completed_process):
        res = health_check.check_pm2()
        assert res["status"] == "error"
        assert "some pm2 error" in res["detail"]

def test_check_pm2_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        res = health_check.check_pm2()
        assert res["status"] == "pm2_not_found"

def test_check_pm2_exception():
    with patch("subprocess.run", side_effect=Exception("generic error")):
        res = health_check.check_pm2()
        assert res["status"] == "error"
        assert "generic error" in res["detail"]

def test_check_state_files(tmp_path):
    # Setup mock state dir
    state_file = tmp_path / "real_state_BTCUSDT.json"
    state_data = {
        "last_tpv": 10500.0,
        "last_profit": 500.0,
        "rebalance_cycles": 42,
        "trailing_stop_triggered": False,
        "tpv_ath": 11000.0,
        "balance": 10000.0
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f)

    with open(state_file, "r", encoding="utf-8") as f:
        print("BTCUSDT content on disk BEFORE calling:", f.read())

    # Invalid state file to test exception handling
    bad_state_file = tmp_path / "real_state_ETHUSDT.json"
    with open(bad_state_file, "w", encoding="utf-8") as f:
        f.write("invalid json")

    with patch("futures_portfolio.health_check.STATE_DIR", tmp_path):
        res = health_check.check_state_files()
        print("Returned state files:", res)
        assert "BTCUSDT" in res
        if "error" in res["BTCUSDT"]:
            print("BTCUSDT error detail:", res["BTCUSDT"]["error"])
        assert res["BTCUSDT"]["tpv"] == 10500.0
        assert res["BTCUSDT"]["cycles"] == 42
        
        assert "ETHUSDT" in res
        assert "error" in res["ETHUSDT"]

def test_check_blacklist(tmp_path):
    config_file = tmp_path / "config.json"
    config_data = {
        "black_list": ["XRPUSDT", "SOLUSDT"],
        "toxic_cooldown_days": 3
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    with patch("futures_portfolio.health_check.STATE_DIR", tmp_path):
        res = health_check.check_blacklist()
        assert res["blacklisted"] == ["XRPUSDT", "SOLUSDT"]
        assert res["toxic_cooldown_days"] == 3

def test_check_blacklist_missing(tmp_path):
    with patch("futures_portfolio.health_check.STATE_DIR", tmp_path):
        res = health_check.check_blacklist()
        assert res["blacklisted"] == []

def test_check_recent_errors(tmp_path):
    # Setup logs folder
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    # Empty or old logs
    ok_log = log_dir / "err_1.log"
    with open(ok_log, "w") as f:
        f.write("some old error line\n")
    
    # Active log with recent modification
    active_log = log_dir / "err_active.log"
    with open(active_log, "w", encoding="utf-8") as f:
        f.write("line 1\nline 2\nline 3\nline 4\nline 5\nline 6\n")

    # Mock os.stat to return recent mtime for active_log, and old mtime for ok_log
    orig_stat = os.stat
    def mock_stat(path, *args, **kwargs):
        path_str = str(path)
        if "err_" in path_str:
            stat_res = orig_stat(path)
            mock_stat_obj = MagicMock()
            mock_stat_obj.st_size = stat_res.st_size
            if "err_active.log" in path_str:
                mock_stat_obj.st_mtime = health_check.time.time() - 300 # 5 minutes ago
            else:
                mock_stat_obj.st_mtime = health_check.time.time() - 7200 # 2 hours ago
            return mock_stat_obj
        return orig_stat(path, *args, **kwargs)

    with patch("futures_portfolio.health_check.LOG_DIR", log_dir):
        with patch("os.stat", side_effect=mock_stat):
            res = health_check.check_recent_errors()
            assert len(res) == 1
            assert res[0]["file"] == "err_active.log"
            assert res[0]["last_errors"] == ["line 2", "line 3", "line 4", "line 5", "line 6"]

def test_main(tmp_path, capsys):
    with patch("futures_portfolio.health_check.STATE_DIR", tmp_path):
        with patch("futures_portfolio.health_check.LOG_DIR", tmp_path):
            with patch("futures_portfolio.health_check.check_pm2", return_value={"status": "ok"}):
                health_check.main()
                captured = capsys.readouterr()
                data = json.loads(captured.out)
                assert "timestamp" in data
                assert data["pm2"] == {"status": "ok"}
