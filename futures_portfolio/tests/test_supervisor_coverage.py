"""
Coverage push: supervisor.py 66% -> 85%+
Targets: get_pm2_processes, reconcile_swarm_state, stop_bot exceptions,
enforce_invariant_gate, reset_bot_state_files, start_bot, _ensure_real_bots_alive,
Amnesty, Reaper Guard, Authoritative Cleanup, manage_swarm real rotation.
"""
import pytest
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

M = "futures_portfolio.supervisor"


# =====================================================================
# get_pm2_processes (lines 85-94)
# =====================================================================

@pytest.mark.asyncio
async def test_get_pm2_processes_empty_stdout():
    """Lines 89-90: empty stdout -> returns []"""
    from futures_portfolio.supervisor import get_pm2_processes
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"")
    with patch(f"{M}.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await get_pm2_processes()
    assert result == []


@pytest.mark.asyncio
async def test_get_pm2_processes_valid_json():
    """Line 91: valid JSON stdout -> parsed list"""
    from futures_portfolio.supervisor import get_pm2_processes
    data = [{"name": "paper-aliceusdt", "pm2_env": {"status": "online"}}]
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (json.dumps(data).encode(), b"")
    with patch(f"{M}.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await get_pm2_processes()
    assert result == data


@pytest.mark.asyncio
async def test_get_pm2_processes_exception():
    """Lines 92-94: exception -> returns []"""
    from futures_portfolio.supervisor import get_pm2_processes
    with patch(f"{M}.asyncio.create_subprocess_exec", side_effect=FileNotFoundError("pm2 not found")):
        result = await get_pm2_processes()
    assert result == []


@pytest.mark.asyncio
async def test_get_pm2_processes_json_decode_error():
    """Line 92: JSON decode error -> returns []"""
    from futures_portfolio.supervisor import get_pm2_processes
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"NOT JSON {{{", b"")
    with patch(f"{M}.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await get_pm2_processes()
    assert result == []


# =====================================================================
# reconcile_swarm_state (lines 96-145)
# =====================================================================

@pytest.mark.asyncio
async def test_reconcile_swarm_state_no_orphans():
    """Lines 103-145: no orphans -> no kills"""
    from futures_portfolio.supervisor import reconcile_swarm_state
    pm2_data = [{"name": "paper-aliceusdt", "pm2_env": {"status": "online"}}]
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (json.dumps(pm2_data).encode(), b"")
    with patch(f"{M}.asyncio.create_subprocess_exec", return_value=mock_proc):
        await reconcile_swarm_state(["ALICEUSDT"], "paper")


@pytest.mark.asyncio
async def test_reconcile_swarm_state_kills_orphans():
    """Lines 117-141: orphaned process -> pm2 delete + pm2 save"""
    from futures_portfolio.supervisor import reconcile_swarm_state
    pm2_data = [
        {"name": "paper-aliceusdt", "pm2_env": {"status": "online"}},
        {"name": "paper-orphonusdt", "pm2_env": {"status": "online"}},
    ]
    call_log = []

    async def fake_exec(cmd, *args, **kwargs):
        call_log.append((cmd, args))
        return AsyncMock()

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (json.dumps(pm2_data).encode(), b"")

    # get_pm2_processes uses create_subprocess_exec, and so does reconcile_swarm_state for delete/save
    with patch(f"{M}.asyncio.create_subprocess_exec", side_effect=fake_exec):
        # Patch get_pm2_processes to return our data directly
        with patch(f"{M}.get_pm2_processes", return_value=pm2_data):
            await reconcile_swarm_state(["ALICEUSDT"], "paper")

    # Should have called pm2 delete for orphan
    delete_calls = [c for c in call_log if c[0] == "pm2" and len(c[1]) > 0 and c[1][0] == "delete"]
    assert len(delete_calls) >= 1
    assert any("orphonusdt" in str(c) for c in delete_calls)


@pytest.mark.asyncio
async def test_reconcile_swarm_state_empty_pm2():
    """Lines 108-115: empty PM2 -> no orphans"""
    from futures_portfolio.supervisor import reconcile_swarm_state
    with patch(f"{M}.get_pm2_processes", return_value=[]):
        await reconcile_swarm_state(["ALICEUSDT"], "paper")


# =====================================================================
# stop_bot exception paths (lines 152-157)
# =====================================================================

@pytest.mark.asyncio
async def test_stop_bot_filenotfound():
    """Lines 152-153: PM2 binary not found -> caught"""
    from futures_portfolio.supervisor import stop_bot
    mock_proc = AsyncMock()
    mock_proc.wait = AsyncMock()
    mock_shell = AsyncMock(return_value=mock_proc)
    # create_subprocess_shell itself raises FileNotFoundError
    with patch(f"{M}.asyncio.create_subprocess_shell", side_effect=FileNotFoundError):
        await stop_bot("BTCUSDT", is_paper=False)


@pytest.mark.asyncio
async def test_stop_bot_oserror():
    """Lines 154-155: OS error -> caught"""
    from futures_portfolio.supervisor import stop_bot
    mock_proc = AsyncMock()
    mock_proc.wait = AsyncMock()
    with patch(f"{M}.asyncio.create_subprocess_shell", side_effect=OSError("permission denied")):
        await stop_bot("ETHUSDT", is_paper=True)


@pytest.mark.asyncio
async def test_stop_bot_general_exception():
    """Lines 156-157: general exception -> caught"""
    from futures_portfolio.supervisor import stop_bot
    mock_proc = AsyncMock()
    mock_proc.wait = AsyncMock()
    with patch(f"{M}.asyncio.create_subprocess_shell", side_effect=RuntimeError("unexpected")):
        await stop_bot("SOLUSDT", is_paper=False)


# =====================================================================
# enforce_invariant_gate (lines 258-284)
# =====================================================================

@pytest.mark.asyncio
async def test_enforce_invariant_gate_adds_exposed_ticker():
    """Lines 263-282: position with notional >= threshold -> forced into live_swarm"""
    from futures_portfolio.supervisor import enforce_invariant_gate
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        "NEWCOINUSDT_LONG": {"qty": "10.0", "mark_price": "10.0"}
    })
    config = {"live_swarm": ["BTCUSDT"], "min_notional_usdt": 5.0}
    await enforce_invariant_gate(connector, config)
    assert "NEWCOINUSDT" in config["live_swarm"]
    assert "BTCUSDT" in config["live_swarm"]


@pytest.mark.asyncio
async def test_enforce_invariant_gate_dust_filtered():
    """Lines 275-276: position below dust threshold -> NOT added"""
    from futures_portfolio.supervisor import enforce_invariant_gate
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        "DUSTUSDT_LONG": {"qty": "0.01", "mark_price": "1.0"}
    })
    config = {"live_swarm": [], "min_notional_usdt": 5.0}
    await enforce_invariant_gate(connector, config)
    assert "DUSTUSDT" not in config["live_swarm"]


@pytest.mark.asyncio
async def test_enforce_invariant_gate_empty_positions():
    """Lines 260-261: no positions -> early return"""
    from futures_portfolio.supervisor import enforce_invariant_gate
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value=None)
    config = {"live_swarm": []}
    await enforce_invariant_gate(connector, config)
    assert config["live_swarm"] == []


@pytest.mark.asyncio
async def test_enforce_invariant_gate_zero_qty_filtered():
    """Line 269: qty=0 -> skip"""
    from futures_portfolio.supervisor import enforce_invariant_gate
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        "ZEROUSDT_LONG": {"qty": "0.0", "mark_price": "100.0"}
    })
    config = {"live_swarm": [], "min_notional_usdt": 5.0}
    await enforce_invariant_gate(connector, config)
    assert "ZEROUSDT" not in config["live_swarm"]


@pytest.mark.asyncio
async def test_enforce_invariant_gate_exception():
    """Lines 283-284: exception -> logged, not raised"""
    from futures_portfolio.supervisor import enforce_invariant_gate
    connector = MagicMock()
    connector.get_positions = AsyncMock(side_effect=RuntimeError("exchange down"))
    config = {"live_swarm": []}
    await enforce_invariant_gate(connector, config)


# =====================================================================
# reset_bot_state_files (lines 286-361)
# =====================================================================

@pytest.mark.asyncio
async def test_reset_bot_state_files_rebases_on_trailing_stop(tmp_path, monkeypatch):
    """Lines 312-313: trailing_stop_triggered -> capital rebased from last_tpv"""
    from futures_portfolio.supervisor import reset_bot_state_files
    monkeypatch.setattr(f"{M}.BASE_PATH", tmp_path)
    old_state = {"trailing_stop_triggered": True, "last_tpv": 200.0}
    (tmp_path / "paper_state_BTCUSDT.json").write_text(json.dumps(old_state))
    config = {"initial_capital": 100.0}
    await reset_bot_state_files("BTCUSDT", is_paper=True, config=config)
    new_state = json.loads((tmp_path / "paper_state_BTCUSDT.json").read_text())
    assert new_state["balance"] == 200.0


@pytest.mark.asyncio
async def test_reset_bot_state_files_no_state_uses_config(tmp_path, monkeypatch):
    """Lines 307-310: no old state -> uses config_capital"""
    from futures_portfolio.supervisor import reset_bot_state_files
    monkeypatch.setattr(f"{M}.BASE_PATH", tmp_path)
    config = {"portfolios": [{"paper_initial_capital": 150.0, "initial_capital": 80.0}]}
    await reset_bot_state_files("ETHUSDT", is_paper=True, config=config)
    new_state = json.loads((tmp_path / "paper_state_ETHUSDT.json").read_text())
    assert new_state["balance"] == 150.0


# =====================================================================
# start_bot (lines 363-378)
# =====================================================================

@pytest.mark.asyncio
async def test_start_bot_real_no_state_file(tmp_path, monkeypatch):
    """Lines 373-375: real bot, no state file -> reset called"""
    from futures_portfolio.supervisor import start_bot
    monkeypatch.setattr(f"{M}.BASE_PATH", tmp_path)
    reset_called = False

    async def fake_reset(ticker, is_paper, config):
        nonlocal reset_called
        reset_called = True

    with patch(f"{M}.reset_bot_state_files", side_effect=fake_reset), \
         patch(f"{M}.asyncio.create_subprocess_shell", return_value=AsyncMock(wait=AsyncMock())):
        await start_bot("BTCUSDT", is_paper=False, config={"initial_capital": 100})
    assert reset_called


@pytest.mark.asyncio
async def test_start_bot_real_with_existing_state(tmp_path, monkeypatch):
    """Lines 373-374: real bot, state exists -> NO reset"""
    from futures_portfolio.supervisor import start_bot
    monkeypatch.setattr(f"{M}.BASE_PATH", tmp_path)
    (tmp_path / "real_state_BTCUSDT.json").write_text("{}")
    reset_called = False

    async def fake_reset(ticker, is_paper, config):
        nonlocal reset_called
        reset_called = True

    with patch(f"{M}.reset_bot_state_files", side_effect=fake_reset), \
         patch(f"{M}.asyncio.create_subprocess_shell", return_value=AsyncMock(wait=AsyncMock())):
        await start_bot("BTCUSDT", is_paper=False, config={"initial_capital": 100})
    assert not reset_called


@pytest.mark.asyncio
async def test_start_bot_paper_always_resets(tmp_path, monkeypatch):
    """Lines 370-371: paper bot -> always resets"""
    from futures_portfolio.supervisor import start_bot
    monkeypatch.setattr(f"{M}.BASE_PATH", tmp_path)
    (tmp_path / "paper_state_BTCUSDT.json").write_text("{}")
    reset_called = False

    async def fake_reset(ticker, is_paper, config):
        nonlocal reset_called
        reset_called = True

    with patch(f"{M}.reset_bot_state_files", side_effect=fake_reset), \
         patch(f"{M}.asyncio.create_subprocess_shell", return_value=AsyncMock(wait=AsyncMock())):
        await start_bot("BTCUSDT", is_paper=True, config={"initial_capital": 100})
    assert reset_called


# =====================================================================
# _ensure_real_bots_alive (lines 1043-1066)
# =====================================================================

@pytest.mark.asyncio
async def test_ensure_real_bots_alive_missing_bot_restarts():
    """Lines 1057-1064: live_swarm bot not in PM2 -> restart"""
    from futures_portfolio.supervisor import _ensure_real_bots_alive
    config = {"live_swarm": ["BTCUSDT"]}
    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.start_bot", AsyncMock()) as mock_start:
        await _ensure_real_bots_alive(config)
    mock_start.assert_called_once_with("BTCUSDT", is_paper=False, config=config)


@pytest.mark.asyncio
async def test_ensure_real_bots_alive_all_running():
    """Lines 1054-1058: all bots running -> no restart"""
    from futures_portfolio.supervisor import _ensure_real_bots_alive
    config = {"live_swarm": ["BTCUSDT"]}
    running = {"r_BTCUSDT": {"name": "real-btcusdt", "paper": False}}
    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value=running)), \
         patch(f"{M}.start_bot", AsyncMock()) as mock_start:
        await _ensure_real_bots_alive(config)
    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_real_bots_alive_empty_swarm():
    """Line 1051: empty live_swarm -> early return"""
    from futures_portfolio.supervisor import _ensure_real_bots_alive
    config = {"live_swarm": []}
    with patch(f"{M}.get_running_bots_info", AsyncMock()) as mock_info:
        await _ensure_real_bots_alive(config)
    mock_info.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_real_bots_alive_start_exception():
    """Line 1065-1066: start_bot raises -> logged, not raised"""
    from futures_portfolio.supervisor import _ensure_real_bots_alive
    config = {"live_swarm": ["BTCUSDT"]}
    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.start_bot", AsyncMock(side_effect=RuntimeError("pm2 crash"))):
        await _ensure_real_bots_alive(config)


# =====================================================================
# enforce_swarm_consistency: exception paths (lines 228-243)
# =====================================================================

@pytest.mark.asyncio
async def test_enforce_swarm_consistency_no_positions():
    """Lines 172-173: no active positions -> return empty set"""
    from futures_portfolio.supervisor import enforce_swarm_consistency
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value=None)
    config = {"live_swarm": [], "real_whitelist": []}
    result = await enforce_swarm_consistency(connector, config)
    assert result == set()


@pytest.mark.asyncio
async def test_enforce_swarm_consistency_heal_failure(tmp_path, monkeypatch):
    """Lines 228-243: orphan with no running PM2 -> attempt heal, exception caught"""
    from futures_portfolio.supervisor import enforce_swarm_consistency
    monkeypatch.setattr(f"{M}.BASE_PATH", tmp_path)
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        "ZECUSDT_LONG": {"qty": "-0.5"}
    })
    config = {"live_swarm": [], "real_whitelist": []}
    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.start_bot", AsyncMock(side_effect=RuntimeError("fail"))):
        result = await enforce_swarm_consistency(connector, config)
    # Should not raise
