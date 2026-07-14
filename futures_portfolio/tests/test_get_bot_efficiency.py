import asyncio
import json
import os
import pytest
from unittest.mock import patch, AsyncMock
from futures_portfolio.supervisor.swarm_manager import get_bot_efficiency
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent / "futures_portfolio"

@pytest.mark.asyncio
async def test_get_bot_efficiency_no_state():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"

    with patch("futures_portfolio.supervisor.swarm_manager.safe_load_json", AsyncMock(return_value={})):
        result = await get_bot_efficiency(ticker, config)

    assert result == {
        "profit": 0.0,
        "cycles": 0,
        "eff": 0.0,
        "trailing_stop_paper_timeout_end": 0.0
    }

@pytest.mark.asyncio
async def test_get_bot_efficiency_with_state():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"
    state = {
        "last_profit": 100.0,
        "rebalance_cycles": 20,
        "trailing_stop_paper_timeout_end": 123456789.0
    }

    with patch("futures_portfolio.supervisor.swarm_manager.safe_load_json", AsyncMock(return_value=state)):
        result = await get_bot_efficiency(ticker, config)

    assert result == {
        "profit": 100.0,
        "cycles": 20,
        "eff": 100.0 / 20,
        "trailing_stop_paper_timeout_end": 123456789.0
    }

@pytest.mark.asyncio
async def test_get_bot_efficiency_min_cycles():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"
    state = {
        "last_profit": 100.0,
        "rebalance_cycles": 5,
        "trailing_stop_paper_timeout_end": 0.0
    }

    with patch("futures_portfolio.supervisor.swarm_manager.safe_load_json", AsyncMock(return_value=state)):
        result = await get_bot_efficiency(ticker, config)

    assert result == {
        "profit": 100.0,
        "cycles": 5,
        "eff": 100.0 / 10, # max(5, 10)
        "trailing_stop_paper_timeout_end": 0.0
    }
