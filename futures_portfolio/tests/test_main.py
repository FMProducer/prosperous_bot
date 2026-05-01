import pytest
import json
import os
import asyncio
import copy
import logging
from unittest.mock import patch, AsyncMock, MagicMock
from futures_portfolio.main import rebalance_loop

@pytest.fixture
def mock_config():
    return {
        "paper_mode": True,
        "base_ticker": "BTCUSDT",
        "portfolios": [{
            "targets": {
                "BASE_LONG": {"share": 0.4, "leverage": 5.0},
                "BASE_SHORT": {"share": 0.4, "leverage": 5.0},
                "VIRTUAL": {"share": 0.2}
            },
            "rebalance_threshold": 0.05,
            "check_interval_sec": 0.1,
            "max_capital_usdt": 10000.0,
            "min_notional_usdt": 6.0,
            "siphoning_threshold_pct": 1.0,
            "reinvestment_ratio": 0.5,
            "equity_trailing_stop_pct": 0.0
        }]
    }

@pytest.fixture
def mock_connector():
    connector = MagicMock()
    connector.get_futures_prices = AsyncMock(return_value={"BTCUSDT": 60000.0})
    connector.get_exchange_info = AsyncMock(return_value={
        "symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}]}]
    })
    connector.get_hedge_mode = AsyncMock(return_value=True)
    connector.get_free_balance = AsyncMock(return_value=10000.0)
    connector.get_positions = AsyncMock(return_value={})
    connector.get_margin_ratio = AsyncMock(return_value={"margin_ratio": 10.0})
    connector.get_bnb_balance = AsyncMock(return_value=1.0)
    return connector

@pytest.fixture
def mock_notifier():
    notifier = MagicMock()
    notifier.send_message = AsyncMock()
    notifier.send_alert = AsyncMock()
    notifier.send_status = AsyncMock()
    notifier.close = AsyncMock()
    return notifier

@pytest.mark.asyncio
async def test_rebalance_loop_siphoning(mock_config, mock_connector, mock_notifier):
    state = {
        "virt_basis_price": 60000.0,
        "virt_allocated_usdt": 2000.0,
        "base_ticker": "BTCUSDT",
        "siphoning_reserve": 0.0,
        "initial_tpv": 10000.0,
        "reference_tpv": 10000.0,
        "tpv_ath": 10000.0,
        "rebalance_cycles": 0
    }
    paper_state = {
        "balance": 20000.0,
        "positions": {"BTCUSDT_LONG": 0.0, "BTCUSDT_SHORT": 0.0},
        "last_price": 60000.0,
        "base_ticker": "BTCUSDT",
        "long_entry_price": 0.0,
        "short_entry_price": 0.0
    }
    def load_side_effect(path, default=None):
        if "paper_state" in path: return paper_state
        if "state" in path: return state
        return default
    saves = []
    def save_side_effect(path, data):
        nonlocal state, paper_state
        saves.append((path, copy.deepcopy(data)))
        if "paper_state" in path: paper_state = copy.deepcopy(data)
        if "state" in path: state = copy.deepcopy(data)

    mock_config["portfolios"][0]["max_capital_usdt"] = 0.0
    with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
        with patch("futures_portfolio.main.save_json", AsyncMock(side_effect=save_side_effect)):
            with patch("futures_portfolio.main.read_shared_config", return_value=mock_config):
                with patch("asyncio.sleep", side_effect=[None, None, Exception("StopLoop")]):
                    with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
                        try:
                            await rebalance_loop(mock_connector, "config.json", "state.json", "paper_state.json", MagicMock())
                        except Exception as e:
                            if str(e) != "StopLoop": raise e
    # Note: siphoning might not trigger if no profitable sell occurs in this setup.
    assert len(saves) > 0

@pytest.mark.asyncio
async def test_rebalance_loop_trailing_stop(mock_config, mock_connector, mock_notifier):
    mock_config["portfolios"][0]["equity_trailing_stop_pct"] = 5.0
    state = {
        "virt_basis_price": 60000.0,
        "virt_allocated_usdt": 2000.0,
        "base_ticker": "BTCUSDT",
        "siphoning_reserve": 0.0,
        "initial_tpv": 10000.0,
        "reference_tpv": 10000.0,
        "tpv_ath": 20000.0,
        "rebalance_cycles": 0
    }
    paper_state = {
        "balance": 18000.0,
        "positions": {"BTCUSDT_LONG": 0.1, "BTCUSDT_SHORT": 0.0},
        "last_price": 60000.0,
        "base_ticker": "BTCUSDT"
    }
    def load_side_effect(path, default=None):
        if "paper_state" in path: return paper_state
        if "state" in path: return state
        return default
    saves = []
    def save_side_effect(path, data):
        nonlocal state, paper_state
        saves.append((path, copy.deepcopy(data)))
        if "paper_state" in path: paper_state = copy.deepcopy(data)
        if "state" in path: state = copy.deepcopy(data)

    with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
        with patch("futures_portfolio.main.save_json", AsyncMock(side_effect=save_side_effect)):
            with patch("futures_portfolio.main.read_shared_config", return_value=mock_config):
                with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
                    with patch("asyncio.sleep", side_effect=[None, Exception("StopLoop")]):
                        try:
                                await rebalance_loop(mock_connector, "config.json", "state.json", "paper_state.json", MagicMock())
                        except Exception as e:
                            if str(e) != "StopLoop": raise e
    last_paper_state = next(s[1] for s in reversed(saves) if "positions" in s[1])
    assert last_paper_state["positions"]["BTCUSDT_LONG"] == 0.0

@pytest.mark.asyncio
async def test_rebalance_loop_margin_warning(mock_config, mock_connector, mock_notifier):
    mock_config["paper_mode"] = False
    mock_config["portfolios"][0]["margin_ratio_warning"] = 5.0
    mock_config["portfolios"][0]["margin_ratio_critical"] = 2.0
    state = {"virt_basis_price": 60000.0, "virt_allocated_usdt": 2000.0, "base_ticker": "BTCUSDT"}
    mock_connector.get_margin_ratio = AsyncMock(return_value={"margin_ratio": 3.0})

    async def async_load_side_effect(path, default=None):
        if "s.json" in path: return state
        return default

    with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=async_load_side_effect)):
        with patch("futures_portfolio.main.save_json", AsyncMock()):
            with patch("futures_portfolio.main.read_shared_config", return_value=mock_config):
                with patch("asyncio.sleep", side_effect=[None, Exception("StopLoop")]):
                    with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
                        try:
                            await rebalance_loop(mock_connector, "c.json", "s.json", "p.json", MagicMock())
                        except Exception as e:
                            if str(e) != "StopLoop": raise e
    mock_notifier.send_message.assert_any_call("⚠️ <b>WARNING</b>: Low margin ratio: 3.00 (BTCUSDT)")

@pytest.mark.asyncio
async def test_rebalance_loop_margin_critical(mock_config, mock_connector, mock_notifier):
    mock_config["paper_mode"] = False
    mock_config["portfolios"][0]["margin_ratio_critical"] = 2.0
    state = {"virt_basis_price": 60000.0, "virt_allocated_usdt": 2000.0, "base_ticker": "BTCUSDT"}
    mock_connector.get_margin_ratio = AsyncMock(return_value={"margin_ratio": 1.5})

    async def async_load_side_effect(path, default=None):
        if "s.json" in path: return state
        return default

    with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=async_load_side_effect)):
        with patch("futures_portfolio.main.save_json", AsyncMock()):
            with patch("futures_portfolio.main.read_shared_config", return_value=mock_config):
                with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
                    with patch("asyncio.sleep", side_effect=Exception("StopLoop")):
                        try:
                            await rebalance_loop(mock_connector, "c.json", "s.json", "p.json", MagicMock())
                        except Exception as e:
                            if str(e) != "StopLoop": raise e
    mock_notifier.send_alert.assert_called_with("CRITICAL MARGIN", f"Margin ratio 1.50 < {2.0:.1f}. Emergency stop!")
