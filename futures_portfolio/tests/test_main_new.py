class ZombieExit(BaseException):
    pass

import pytest
import json
import os
import asyncio
import copy
import logging
from decimal import Decimal
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
from futures_portfolio.core.main import rebalance_loop, emit_signal
from pathlib import Path

@pytest.fixture
def mock_config():
    return {
        "paper_mode": True,
        "base_ticker": "BTCUSDT",
        "equity_trailing_stop_pct": 5.0,
        "equity_trailing_stop_timeout_sec": 0.0,
        "portfolios": [{
            "initial_capital": 10000.0,
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
            "equity_trailing_stop_pct": 5.0,
            "margin_ratio_warning": 5.0,
            "margin_ratio_critical": 2.0
        }]
    }

@pytest.fixture
def mock_connector():
    connector = MagicMock()
    connector.get_futures_prices = AsyncMock(return_value={"BTCUSDT": 60000.0})
    connector.get_mark_prices = AsyncMock(return_value={"BTCUSDT": 60000.0})
    connector.get_exchange_info = AsyncMock(return_value={
        "symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}]}]
    })
    connector.get_hedge_mode = AsyncMock(return_value=True)
    connector.get_free_balance = AsyncMock(return_value=10000.0)
    connector.get_positions = AsyncMock(return_value={"BTCUSDT_LONG": {"qty": 1.0, "entry_price": 60000.0}})
    connector.get_margin_ratio = AsyncMock(return_value={"margin_ratio": 10.0, "total_wallet_balance": 10000.0})
    connector.get_bnb_balance = AsyncMock(return_value=1.0)
    connector.get_order_book = AsyncMock(return_value={"bids": [["60000", "1"]], "asks": [["60001", "1"]]})
    connector.set_leverage = AsyncMock()
    connector.set_margin_type = AsyncMock()
    return connector

@pytest.fixture
def mock_notifier():
    notifier = MagicMock()
    notifier.send_message = AsyncMock()
    notifier.send_alert = AsyncMock()
    notifier.send_status = AsyncMock()
    notifier.close = AsyncMock()
    return notifier

def sleep_side_effect(secs, *args):
    if secs == 86400:
        raise ZombieExit("ZombieMode")
    return None

def make_bounded_sleep(max_loops=10):
    loop_count = 0
    def bounded_sleep(secs, *args):
        nonlocal loop_count
        if secs == 86400:
            raise ZombieExit("ZombieMode")
        loop_count += 1
        if loop_count > max_loops:
            raise Exception("StopLoop")
        return None
    return bounded_sleep

@pytest.mark.asyncio
async def test_rebalance_loop_siphoning(mock_config, mock_connector, mock_notifier):
    state = {
        "virt_qty": 0.033333,
        "base_ticker": "BTCUSDT",
        "siphoning_reserve": 0.0,
        "initial_tpv": 10000.0,
        "reference_tpv": 10000.0,
        "tpv_ath": 10000.0,
        "rebalance_cycles": 10
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
        if "paper_state" in path: return copy.deepcopy(paper_state)
        if "state" in path: return copy.deepcopy(state)
        return default

    saves = []
    def save_side_effect(path, data):
        saves.append((path, copy.deepcopy(data)))

    mock_config["portfolios"][0]["max_capital_usdt"] = 0.0

    # We want to stop the loop after a few iterations
    loop_count = 0
    def bounded_sleep(secs, *args):
        nonlocal loop_count
        if secs == 86400: raise ZombieExit("ZombieMode")
        loop_count += 1
        if loop_count > 5: raise Exception("StopLoop")
        return None

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.core.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.core.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.core.main.save_json", AsyncMock(side_effect=save_side_effect)):
                with patch("futures_portfolio.core.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=bounded_sleep):
                            with patch("futures_portfolio.core.main.TelegramNotifier", return_value=mock_notifier):
                                try:
                                    await rebalance_loop(mock_connector, "config.json", "state.json", "paper_state.json", MagicMock())
                                except (Exception, BaseException) as e:
                                    if str(e) not in ["StopLoop", "ProcessExit", "ZombieMode"]: raise e
    assert len(saves) > 0

@pytest.mark.asyncio
async def test_rebalance_loop_trailing_stop(mock_config, mock_connector, mock_notifier):
    mock_config["equity_trailing_stop_pct"] = 5.0
    mock_config["equity_trailing_stop_timeout_sec"] = 0.0

    state = {
        "virt_qty": 0.033333,
        "base_ticker": "BTCUSDT",
        "siphoning_reserve": 0.0,
        "initial_tpv": 10000.0,
        "reference_tpv": 10000.0,
        "tpv_ath": 100000.0,
        "rebalance_cycles": 10
    }
    paper_state = {
        "balance": 18000.0,
        "positions": {"BTCUSDT_LONG": 0.1, "BTCUSDT_SHORT": 0.0},
        "last_price": 60000.0,
        "base_ticker": "BTCUSDT",
        "long_entry_price": 60000.0,
        "short_entry_price": 60000.0
    }
    def load_side_effect(path, default=None):
        if "paper_state" in path: return copy.deepcopy(paper_state)
        if "state" in path: return copy.deepcopy(state)
        return default
    saves = []
    def save_side_effect(path, data):
        saves.append((path, copy.deepcopy(data)))

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.core.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.core.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.core.main.save_json", AsyncMock(side_effect=save_side_effect)):
                with patch("futures_portfolio.core.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("futures_portfolio.core.main.TelegramNotifier", return_value=mock_notifier):
                        with patch("futures_portfolio.core.main.self_kill_pm2", AsyncMock()):
                            with patch("asyncio.sleep", side_effect=make_bounded_sleep(max_loops=10)):
                                try:
                                        await rebalance_loop(mock_connector, "config.json", "state.json", "paper_state.json", MagicMock())
                                except (Exception, BaseException) as e:
                                    if str(e) not in ["StopLoop", "ProcessExit", "ZombieMode"]: raise e

    # Check if any paper_state save has 0 positions
    reset_save = next((s[1] for s in saves if "paper_state" in s[0] and s[1]["positions"].get("BTCUSDT_LONG") == 0.0), None)
    assert reset_save is not None

@pytest.mark.asyncio
async def test_rebalance_loop_margin_warning(mock_config, mock_connector, mock_notifier):
    mock_config["paper_mode"] = False
    state = {"virt_qty": 0.0, "base_ticker": "BTCUSDT", "initial_tpv": 10000.0, "rebalance_cycles": 10}
    mock_connector.get_margin_ratio = AsyncMock(return_value={"margin_ratio": 3.0, "total_wallet_balance": 10000.0})
    mock_connector.get_positions = AsyncMock(return_value={"BTCUSDT_LONG": {"qty": 1.0, "entry_price": 60000.0}})

    def load_side_effect(path, default=None):
        if "s.json" in path: return copy.deepcopy(state)
        return default

    loop_count = 0
    def bounded_sleep(secs, *args):
        nonlocal loop_count
        if secs == 86400: raise ZombieExit("ZombieMode")
        loop_count += 1
        if loop_count > 5: raise Exception("StopLoop")
        return None

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.core.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.core.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.core.main.save_json", AsyncMock()):
                with patch("futures_portfolio.core.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=bounded_sleep):
                            with patch("futures_portfolio.core.main.TelegramNotifier", return_value=mock_notifier):
                                try:
                                    await rebalance_loop(mock_connector, "c.json", "s.json", "p.json", MagicMock())
                                except (Exception, BaseException) as e:
                                    if str(e) not in ["StopLoop", "ProcessExit", "ZombieMode"]: raise e
    mock_notifier.send_message.assert_any_call("⚠️ <b>WARNING</b>: Low margin ratio: 3.00 (BTCUSDT)")

@pytest.mark.asyncio
async def test_rebalance_loop_margin_critical(mock_config, mock_connector, mock_notifier):
    mock_config["paper_mode"] = False
    state = {"virt_qty": 0.0, "base_ticker": "BTCUSDT", "initial_tpv": 10000.0, "rebalance_cycles": 10}
    mock_connector.get_margin_ratio = AsyncMock(return_value={"margin_ratio": 1.5, "total_wallet_balance": 10000.0})
    mock_connector.get_positions = AsyncMock(return_value={"BTCUSDT_LONG": {"qty": 1.0, "entry_price": 60000.0}})

    def load_side_effect(path, default=None):
        if "s.json" in path: return copy.deepcopy(state)
        return default

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.core.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.core.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.core.main.save_json", AsyncMock()):
                with patch("futures_portfolio.core.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("futures_portfolio.core.main.self_kill_pm2", AsyncMock()):
                        with patch("asyncio.sleep", side_effect=make_bounded_sleep(max_loops=10)):
                            with patch("futures_portfolio.core.main.TelegramNotifier", return_value=mock_notifier):
                                try:
                                    await rebalance_loop(mock_connector, "c.json", "s.json", "p.json", MagicMock())
                                except (Exception, BaseException) as e:
                                    if str(e) not in ["StopLoop", "ProcessExit", "ZombieMode"]: raise e
    mock_notifier.send_alert.assert_called_with("CRITICAL MARGIN", "Margin ratio 1.50 < 2.0. Emergency stop!")

@pytest.mark.asyncio
async def test_clean_slate_protocol_activation(mock_config, mock_connector, mock_notifier):
    mock_config["paper_mode"] = False
    mock_config["portfolios"][0]["initial_capital"] = 100.0

    state = {
        "virt_qty": 1.5,
        "tpv_ath": 600.0,
        "rebalance_cycles": 100,
        "base_ticker": "BTCUSDT",
        "initial_tpv": 500.0
    }
    paper_state = {
        "balance": 500.0,
        "positions": {"BTCUSDT_LONG": 0.0, "BTCUSDT_SHORT": 0.0},
        "base_ticker": "BTCUSDT"
    }

    mock_connector.get_positions = AsyncMock(return_value={}) # Empty

    def load_side_effect(path, default=None):
        if "p.json" in path: return copy.deepcopy(paper_state)
        if "s.json" in path: return copy.deepcopy(state)
        return default

    saves = []
    def save_side_effect(path, data):
        saves.append((path, copy.deepcopy(data)))

    loop_count = 0
    def bounded_sleep(secs, *args):
        nonlocal loop_count
        if secs == 86400: raise ZombieExit("ZombieMode")
        loop_count += 1
        if loop_count > 1: raise Exception("StopLoop")
        return None

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.core.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.core.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.core.main.save_json", AsyncMock(side_effect=save_side_effect)):
                with patch("futures_portfolio.core.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=bounded_sleep):
                            with patch("futures_portfolio.core.main.TelegramNotifier", return_value=mock_notifier):
                                try:
                                    await rebalance_loop(mock_connector, "c.json", "s.json", "p.json", MagicMock())
                                except (Exception, BaseException) as e:
                                    if str(e) not in ["StopLoop", "ProcessExit", "ZombieMode"]: raise e

    # Check if a save happened with reset values (the protocol saves them immediately)
    reset_state_save = next((s[1] for s in saves if "s.json" in s[0] and s[1]["tpv_ath"] == 100.0), None)
    reset_paper_save = next((s[1] for s in saves if "p.json" in s[0] and s[1]["balance"] == 100.0), None)

    assert reset_state_save is not None
    assert reset_paper_save is not None
    assert reset_state_save["virt_qty"] == 0.0
    assert reset_state_save["rebalance_cycles"] == 0

def test_emit_signal_file_naming(monkeypatch):
    """Подтверждает строгую изоляцию генерации сигналов между Paper и Real режимами."""
    called_paths = []

    def mock_touch(self, *args, **kwargs):
        called_paths.append(self.name)

    monkeypatch.setattr(Path, "touch", mock_touch)
    monkeypatch.setattr(Path, "mkdir", MagicMock())

    emit_signal("stop", "BTCUSDT", is_paper=True)
    emit_signal("exit", "ETHUSDT", is_paper=False)

    assert "stop_paper_BTCUSDT.flag" in called_paths
    assert "exit_real_ETHUSDT.flag" in called_paths

@pytest.mark.asyncio
async def test_self_kill_pm2_flow():
    from futures_portfolio.core.main import self_kill_pm2
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    with patch("asyncio.create_subprocess_shell", return_value=mock_proc) as mock_shell:
        await self_kill_pm2("BTCUSDT", is_paper=True)
        mock_shell.assert_called_once_with(
            "pm2 delete paper-btc",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

@pytest.mark.asyncio
async def test_update_final_metrics_for_exit_flow(tmp_path):
    from futures_portfolio.core.main import _update_final_metrics_for_exit
    state_file = tmp_path / "state.json"
    state = {"initial_tpv": 100.0, "reference_tpv": 100.0}
    safe_calc_res = {"total_pnl_pct": 5.5}
    
    with patch("futures_portfolio.core.main.save_json", AsyncMock()) as mock_save:
        await _update_final_metrics_for_exit(
            state, str(state_file), Decimal("105.0"), Decimal("100.0"), safe_calc_res, 42, MagicMock()
        )
        assert state["last_tpv"] == 105.0
        assert state["rebalance_cycles"] == 42
        assert state["total_pnl_pct"] == 5.5
        mock_save.assert_called_once_with(str(state_file), state)

@pytest.mark.asyncio
async def test_handle_liquidation_recovery_flow(tmp_path):
    from futures_portfolio.core.main import _handle_liquidation_recovery
    
    connector = MagicMock()
    connector.get_position_risk = AsyncMock(return_value={
        "BTCUSDT_LONG": {"qty": 1.5, "unrealized_pnl": -10.0, "positionAmt": 1.5},
        "BTCUSDT_SHORT": {"qty": 0.0, "unrealized_pnl": 0.0, "positionAmt": 0.0}
    })
    
    state_file = tmp_path / "state.json"
    paper_state_file = tmp_path / "paper_state.json"
    config_file = tmp_path / "config.json"
    
    config_data = {
        "live_swarm": ["BTCUSDT", "ETHUSDT"],
        "toxic_blacklist_real": {}
    }
    with open(config_file, "w") as f:
        json.dump(config_data, f)
        
    state = {"initial_tpv": 100.0}
    paper_state = {}
    
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()
    
    with patch("futures_portfolio.core.main.save_json", AsyncMock()) as mock_save:
        with patch("futures_portfolio.core.main.emit_signal") as mock_signal:
            await _handle_liquidation_recovery(
                connector, "BTCUSDT", state, str(state_file),
                paper_state, str(paper_state_file), str(config_file),
                MagicMock(), notifier
            )
            assert state["total_pnl_pct"] == -100.0
            assert state["trailing_stop_triggered"] is True
            mock_save.assert_called_once_with(str(state_file), state)
            # [SSOT] main.py does NOT mutate config.json — only emits signal
            mock_signal.assert_called_with("stop", "BTCUSDT", is_paper=False)

@pytest.mark.asyncio
async def test_rebalance_loop_zombie_on_startup(mock_config, mock_connector, mock_notifier):
    # Tests that if state has trailing_stop_triggered=True, loop exits to Zombie Mode immediately
    state = {"trailing_stop_triggered": True}
    
    def load_side_effect(path, default=None):
        if "s.json" in path or "state" in path: return state
        return default

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))), \
         patch("futures_portfolio.core.main.safe_load_json_sync", return_value=mock_config), \
         patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.core.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.core.main.save_json", AsyncMock()):
                with patch("futures_portfolio.core.main.self_kill_pm2", AsyncMock()) as mock_kill:
                    with patch("asyncio.sleep", side_effect=ZombieExit("Zombie")):
                        try:
                            await rebalance_loop(mock_connector, "c.json", "s.json", "p.json", MagicMock())
                        except ZombieExit:
                            pass
                        mock_kill.assert_called_once_with("BTCUSDT", True)

@pytest.mark.asyncio
async def test_handle_liquidation_guard_paper_critical():
    from futures_portfolio.core.main import _handle_liquidation_guard
    paper_state = {
        "balance": 100.0,
        "positions": {"BTCUSDT_LONG": 1.0, "BTCUSDT_SHORT": 1.0},
        "long_entry_price": 50.0,
        "short_entry_price": 50.0
    }
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()
    logger = MagicMock()
    
    await _handle_liquidation_guard(
        pos_key="BTCUSDT_LONG", dist=4.0, liq_price=45.0,
        liquidation_distance_warn=10.0, liquidation_distance_crit=5.0,
        is_paper=True, raw_positions={}, paper_state=paper_state,
        connector=MagicMock(), base_ticker="BTCUSDT", step_sizes={},
        notifier=notifier, logger=logger
    )
    # 1.0 * (45.0 - 50.0) = -5.0 PnL for Long
    # 1.0 * (50.0 - 45.0) = -5.0? No, Short PnL = 1 * (50 - 45) = +5.0 PnL
    # So balance = 100.0 - 5.0 + 5.0 = 100.0
    assert paper_state["balance"] == 100.0
    assert paper_state["positions"]["BTCUSDT_LONG"] == 0.0
    assert paper_state["positions"]["BTCUSDT_SHORT"] == 0.0

@pytest.mark.asyncio
async def test_handle_liquidation_guard_real_critical():
    from futures_portfolio.core.main import _handle_liquidation_guard
    
    raw_positions = {
        "BTCUSDT_LONG": {"qty": 1.5},
        "BTCUSDT_SHORT": {"qty": -1.5}
    }
    connector = MagicMock()
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()
    logger = MagicMock()
    
    with patch("futures_portfolio.core.main.PortfolioExecutor") as mock_exec_cls:
        mock_executor = MagicMock()
        mock_executor.execute_market_order = AsyncMock()
        mock_exec_cls.return_value = mock_executor
        
        await _handle_liquidation_guard(
            pos_key="BTCUSDT_LONG", dist=4.0, liq_price=45.0,
            liquidation_distance_warn=10.0, liquidation_distance_crit=5.0,
            is_paper=False, raw_positions=raw_positions, paper_state={},
            connector=connector, base_ticker="BTCUSDT", step_sizes={"BTCUSDT": 0.1},
            notifier=notifier, logger=logger
        )
        assert mock_executor.execute_market_order.call_count == 2

@pytest.mark.asyncio
async def test_handle_liquidation_guard_warning():
    from futures_portfolio.core.main import _handle_liquidation_guard
    notifier = MagicMock()
    logger = MagicMock()
    
    await _handle_liquidation_guard(
        pos_key="BTCUSDT_LONG", dist=8.0, liq_price=45.0,
        liquidation_distance_warn=10.0, liquidation_distance_crit=5.0,
        is_paper=True, raw_positions={}, paper_state={},
        connector=MagicMock(), base_ticker="BTCUSDT", step_sizes={},
        notifier=notifier, logger=logger
    )
    logger.warning.assert_called_once()
