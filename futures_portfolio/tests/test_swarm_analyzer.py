import pytest
import io
import sys
import os
import json
from unittest.mock import MagicMock, patch
from futures_portfolio.supervisor.swarm_analyzer import analyze_swarm


def test_analyze_swarm():
    config = {
        "portfolios": [{"initial_capital": 1000.0}],
        "max_bots": 2,
        "tickers": ["BTCUSDT"],
        "live_swarm": ["BTCUSDT"]
    }

    state_data = {
        "base_ticker": "BTCUSDT",
        "last_profit": 100.0,
        "rebalance_cycles": 50,
        "siphoning_reserve": 10.0
    }

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        with patch("futures_portfolio.supervisor.swarm_analyzer.safe_load_json_sync", side_effect=[config, state_data, state_data]), \
             patch("futures_portfolio.supervisor.swarm_analyzer.glob.glob", return_value=["real_state_BTCUSDT.json"]), \
             patch("os.path.exists", return_value=True):
            analyze_swarm()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "BTCUSDT" in output
    assert "100.00" in output
    assert "ROI: 5.00%" in output  # (100 / 2000) * 100


def test_analyze_swarm_missing_config():
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        with patch("futures_portfolio.supervisor.swarm_analyzer.safe_load_json_sync", return_value={}), \
             patch("futures_portfolio.supervisor.swarm_analyzer.glob.glob", return_value=[]):
            analyze_swarm()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "Error loading config.json" in output
