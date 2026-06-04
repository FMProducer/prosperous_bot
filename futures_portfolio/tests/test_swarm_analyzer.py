import pytest
import os
import json
from unittest.mock import MagicMock, patch
from swarm_analyzer import analyze_swarm

def test_analyze_swarm(capsys):
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
    
    with patch("swarm_analyzer.safe_load_json_sync", side_effect=[config, state_data, state_data]), \
         patch("swarm_analyzer.glob.glob", return_value=["real_state_BTCUSDT.json"]), \
         patch("os.path.exists", return_value=True):
        
        analyze_swarm()
        
        captured = capsys.readouterr()
        assert "BTCUSDT" in captured.out
        assert "100.00" in captured.out
        assert "ROI: 5.00%" in captured.out # (100 / 2000) * 100

def test_analyze_swarm_missing_config(capsys):
    with patch("swarm_analyzer.safe_load_json_sync", return_value={}), \
         patch("swarm_analyzer.glob.glob", return_value=[]):
        
        analyze_swarm()
        captured = capsys.readouterr()
        assert "Error loading config.json" in captured.out
