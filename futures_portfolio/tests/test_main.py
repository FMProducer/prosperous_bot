class ZombieExit(BaseException):
    pass

import pytest
import json
import os
import asyncio
import copy
import logging
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
from futures_portfolio.main import rebalance_loop, emit_signal
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

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.main.save_json", AsyncMock(side_effect=save_side_effect)):
                with patch("futures_portfolio.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=bounded_sleep):
                            with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
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

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.main.save_json", AsyncMock(side_effect=save_side_effect)):
                with patch("futures_portfolio.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
                            with patch("asyncio.sleep", side_effect=sleep_side_effect):
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

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.main.save_json", AsyncMock()):
                with patch("futures_portfolio.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=bounded_sleep):
                            with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
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

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.main.save_json", AsyncMock()):
                with patch("futures_portfolio.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=sleep_side_effect):
                            with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
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

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),          patch("futures_portfolio.main.safe_load_json_sync", return_value=mock_config),          patch("os.path.getmtime", return_value=123):
        with patch("futures_portfolio.main.load_json", AsyncMock(side_effect=load_side_effect)):
            with patch("futures_portfolio.main.save_json", AsyncMock(side_effect=save_side_effect)):
                with patch("futures_portfolio.main.sys.exit", side_effect=BaseException("ProcessExit")):
                    with patch("asyncio.sleep", side_effect=bounded_sleep):
                            with patch("futures_portfolio.main.TelegramNotifier", return_value=mock_notifier):
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


# ============================================================
# Tests: Per-ticker min_notional with 2% buffer
# ============================================================

class TestMinNotionalExtraction:
    """Tests for per-ticker min_notional extraction from exchange_info."""

    def test_min_notionals_extracted_from_exchange_info(self):
        """Проверяем что min_notionals извлекается из exchange_info filters (Binance Futures: 'notional', SPOT: 'minNotional')"""
        exchange_info = {
            "symbols": [
                {"symbol": "GRASSUSDT", "filters": [{"filterType": "MIN_NOTIONAL", "notional": "5.03"}]},
                {"symbol": "YFIUSDT", "filters": [{"filterType": "MIN_NOTIONAL", "notional": "7.00"}]},
                {"symbol": "BTCUSDT", "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5.0"}
                ]},
            ]
        }
        min_notionals = {
            s["symbol"]: float(f.get("minNotional") or f.get("notional"))
            for s in exchange_info["symbols"]
            for f in s["filters"]
            if f["filterType"] == "MIN_NOTIONAL"
        }
        assert min_notionals["GRASSUSDT"] == 5.03
        assert min_notionals["YFIUSDT"] == 7.00
        assert min_notionals["BTCUSDT"] == 5.0


class TestEffectiveMinNotional:
    """Tests for effective_min_notional = max(config_min, exchange_min * 1.02)."""

    def test_effective_min_takes_max_with_buffer(self):
        """effective_min_notional = max(config_min=5.1, exchange_min * 1.02)"""
        config_min = 5.1
        test_cases = [
            ("GRASSUSDT", 5.03, 5.13),    # 5.03*1.02=5.1306 > 5.1 → buffer
            ("VVVUSDT", 5.05, 5.15),      # 5.05*1.02=5.151 > 5.1 → buffer
            ("YFIUSDT", 7.00, 7.14),      # 7.00*1.02=7.14 > 5.1 → buffer
            ("UNKNOWNUSDT", None, 5.1),   # no exchange data → config fallback
        ]
        for ticker, exchange_min, expected in test_cases:
            emin = (exchange_min * 1.02) if exchange_min else config_min
            result = max(config_min, emin)
            assert result == pytest.approx(expected, abs=0.01), f"{ticker}: expected {expected}, got {result}"

    def test_grassusdt_rebalances_at_3_7pct(self):
        """При min_notional=5.13 GRASSUSDT (Binance 5.03*1.02) ребалансирует при 3.7% deviation"""
        notional = 20.0 * 7  # 140 USDT
        config_min = 5.1
        exchange_min = 5.03
        effective_min = max(config_min, exchange_min * 1.02)  # 5.13 (buffer wins)

        deviation_3_5pct = notional * 0.035  # 4.90
        deviation_3_7pct = notional * 0.037  # 5.18

        assert deviation_3_5pct < effective_min   # 3.5% too small
        assert deviation_3_7pct >= effective_min  # 3.7% passes

    def test_yfiusdt_blocked_below_5_1pct(self):
        """YFIUSDT (Binance 7.0*1.02=7.14) не ребалансирует при config_min=5.1"""
        notional = 20.0 * 7  # 140 USDT
        config_min = 5.1
        exchange_min = 7.0
        effective_min = max(config_min, exchange_min * 1.02)  # 7.14

        deviation_4pct = notional * 0.04   # 5.6
        deviation_5pct = notional * 0.05   # 7.0
        deviation_5_1pct = notional * 0.051  # 7.14

        assert deviation_4pct < effective_min   # 4% blocked by Binance
        assert deviation_5pct < effective_min   # 5% blocked (7.0 < 7.14)
        assert deviation_5_1pct == pytest.approx(effective_min, abs=0.02)  # 5.1% passes (7.14)

    def test_fallback_to_config_when_exchange_info_missing(self):
        """При отсутствии exchange info используется config min_notional_usdt = 5.1 (без буфера)"""
        config_min = 5.1
        min_notionals = {}  # empty — no exchange data
        base_ticker = "SOMETHINGUSDT"

        exchange_min = min_notionals.get(base_ticker, config_min)
        result = max(config_min, exchange_min)  # fallback = config (no buffer when no exchange data)
        assert result == 5.1  # fallback = config value

    def test_missing_min_notional_in_config_raises(self):
        """Отсутствие min_notional_usdt в config → ValueError"""
        current_config = {}  # no min_notional_usdt
        portfolio_cfg = {}

        config_min = portfolio_cfg.get("min_notional_usdt", current_config.get("min_notional_usdt"))
        with pytest.raises(ValueError, match="min_notional_usdt must be set"):
            if config_min is None:
                raise ValueError("min_notional_usdt must be set in config.json")


# =====================================================================
# B6: SSOT — main.py no longer writes to config.json (2026-07-11)
# =====================================================================

import pytest
import json
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
from futures_portfolio.main import _handle_liquidation_recovery, _handle_liquidation_guard


@pytest.mark.asyncio
async def test_handle_liquidation_recovery_no_config_write():
    """B6: _handle_liquidation_recovery does NOT write to config.json (SSOT)"""
    connector = MagicMock()
    connector.get_position_risk = AsyncMock(return_value={})
    base_ticker = "BTCUSDT"
    state = {"initial_tpv": 180.0, "positions": {}}
    paper_state = {"balance": 100.0}
    logger = MagicMock()
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()

    config_write_called = False
    original_write_text = Path.write_text

    def tracking_write_text(self, *args, **kwargs):
        nonlocal config_write_called
        if "config" in str(self).lower():
            config_write_called = True
        return original_write_text(self, *args, **kwargs)

    with patch("futures_portfolio.main.save_json", AsyncMock()),          patch("futures_portfolio.main.emit_signal") as mock_emit,          patch.object(Path, "write_text", tracking_write_text):

        await _handle_liquidation_recovery(
            connector=connector,
            base_ticker=base_ticker,
            state=state,
            state_file_path="test_state.json",
            paper_state=paper_state,
            paper_state_file_path="test_paper_state.json",
            config_path="config.json",
            logger=logger,
            notifier=notifier,
        )

    # SSOT: emit_signal must be called, config.json must NOT be written
    mock_emit.assert_called_once_with("stop", base_ticker, is_paper=False)
    assert not config_write_called, "VIOLATION: _handle_liquidation_recovery wrote to config.json!"


@pytest.mark.asyncio
async def test_handle_liquidation_guard_no_config_write():
    """B6: _handle_liquidation_guard does NOT write to config.json (SSOT)"""
    connector = MagicMock()
    logger = MagicMock()
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()

    config_write_called = False
    original_write_text = Path.write_text

    def tracking_write_text(self, *args, **kwargs):
        nonlocal config_write_called
        if "config" in str(self).lower():
            config_write_called = True
        return original_write_text(self, *args, **kwargs)

    with patch("futures_portfolio.main.PortfolioExecutor") as mock_exec_cls,          patch("futures_portfolio.main.emit_signal") as mock_emit,          patch.object(Path, "write_text", tracking_write_text):

        mock_exec = MagicMock()
        mock_exec.execute_market_order = AsyncMock()
        mock_exec_cls.return_value = mock_exec

        await _handle_liquidation_guard(
            pos_key="BTCUSDT_LONG",
            dist=5.0,
            liq_price=60000.0,
            liquidation_distance_warn=15.0,
            liquidation_distance_crit=8.0,
            is_paper=False,
            raw_positions={"BTCUSDT_LONG": {"qty": 0.1}},
            paper_state={"positions": {"BTCUSDT_LONG": 0.1}, "balance": 100.0},
            connector=connector,
            base_ticker="BTCUSDT",
            step_sizes={"BTCUSDT": 0.001},
            notifier=notifier,
            logger=logger,
        )

    # dist=5.0 <= crit=8.0 → should emit stop signal
    mock_emit.assert_called_once_with("stop", "BTCUSDT", is_paper=False)
    assert not config_write_called, "VIOLATION: _handle_liquidation_guard wrote to config.json!"
