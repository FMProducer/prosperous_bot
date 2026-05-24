import asyncio
import json
import os
import pytest
from unittest.mock import patch, AsyncMock
from futures_portfolio.supervisor import get_bot_efficiency
from pathlib import Path
from decimal import Decimal

BASE_PATH = Path(__file__).resolve().parent.parent / "futures_portfolio"

@pytest.mark.asyncio
async def test_get_bot_efficiency_no_state():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"

    with patch("futures_portfolio.supervisor.safe_load_json", AsyncMock(return_value={})):
        result = await get_bot_efficiency(ticker, config)

    assert result == {
        "profit": 0.0,
        "tgv": 0.0,
        "cycles": 0,
        "eff": 0.0,
        "pure_eff": 0.0,
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

    with patch("futures_portfolio.supervisor.safe_load_json", AsyncMock(return_value=state)):
        result = await get_bot_efficiency(ticker, config)

    assert result == {
        "profit": 100.0,
        "tgv": 100.0,
        "cycles": 20,
        "eff": 100.0 / 20,
        "pure_eff": 100.0 / 20,
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

    with patch("futures_portfolio.supervisor.safe_load_json", AsyncMock(return_value=state)):
        result = await get_bot_efficiency(ticker, config)

    assert result == {
        "profit": 100.0,
        "tgv": 100.0,
        "cycles": 5,
        "eff": 100.0 / 10, # max(5, 10)
        "pure_eff": 100.0 / 10,
        "trailing_stop_paper_timeout_end": 0.0
    }

@pytest.mark.asyncio
async def test_get_bot_efficiency_with_siphoning():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"
    state = {
        "last_profit": 100.0,
        "siphoning_reserve": 50.0,
        "rebalance_cycles": 10,
        "trailing_stop_paper_timeout_end": 0.0
    }

    with patch("futures_portfolio.supervisor.safe_load_json", AsyncMock(return_value=state)):
        result = await get_bot_efficiency(ticker, config)

    # tgv = 100 + 50 = 150
    # pure_eff = 150 / 10 = 15
    assert result["tgv"] == 150.0
    assert result["pure_eff"] == 15.0
    assert result["eff"] == 15.0

@pytest.mark.asyncio
async def test_get_bot_efficiency_with_drawdown():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"
    state = {
        "last_profit": 100.0,
        "rebalance_cycles": 10,
        "tpv_ath": 1000.0,
        "last_tpv": 900.0, # 10% drawdown
        "trailing_stop_paper_timeout_end": 0.0
    }

    with patch("futures_portfolio.supervisor.safe_load_json", AsyncMock(return_value=state)):
        result = await get_bot_efficiency(ticker, config)

    # tgv = 100
    # pure_eff = 100 / 10 = 10
    # drawdown_pct = (1000 - 900) / 1000 * 100 = 10
    # dd_penalty = 1 + 10 / 100 = 1.1
    # risk_adjusted_eff = 10 / 1.1 = 9.090909...
    assert result["pure_eff"] == 10.0
    assert pytest.approx(result["eff"]) == 10.0 / 1.1
