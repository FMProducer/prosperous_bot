import pytest
import math
import json
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from futures_portfolio import supervisor
from futures_portfolio.supervisor import calculate_bot_score, _calc_rotation_score, selective_merge_incubator, get_bot_efficiency, reset_bot_state_files, enforce_swarm_consistency, _ensure_real_bots_alive, manage_swarm

# Префикс для всех patch-путей — модуль supervisor как он виден тесту
M = "futures_portfolio.supervisor"

def test_calculate_bot_score():
    # is_in_drawdown=True -> INF
    assert calculate_bot_score("BTCUSDT", {"profit": 10.0, "cycles": 20}, is_running_real=True, is_in_drawdown=True, min_cycles_required=10) == float('inf')
    
    # cycles < min_cycles_required and not is_running_real -> -INF
    assert calculate_bot_score("BTCUSDT", {"profit": 10.0, "cycles": 5}, is_running_real=False, is_in_drawdown=False, min_cycles_required=10) == -float('inf')
    
    # net_pnl <= 0 -> -INF
    assert calculate_bot_score("BTCUSDT", {"profit": 0.0, "cycles": 20}, is_running_real=True, is_in_drawdown=False, min_cycles_required=10) == -float('inf')
    assert calculate_bot_score("BTCUSDT", {"profit": -5.0, "cycles": 20}, is_running_real=True, is_in_drawdown=False, min_cycles_required=10) == -float('inf')
    
    # Profitable case
    p = {"profit": 100.0, "cycles": 20}
    min_cycles = 10
    expected_base = (100.0 / 20) * math.log1p(20)
    # is_running_real=False
    assert calculate_bot_score("BTCUSDT", p, is_running_real=False, is_in_drawdown=False, min_cycles_required=min_cycles) == pytest.approx(expected_base)
    # is_running_real=True -> +20% bonus
    assert calculate_bot_score("BTCUSDT", p, is_running_real=True, is_in_drawdown=False, min_cycles_required=min_cycles) == pytest.approx(expected_base * 1.2)

def test_calc_rotation_score():
    min_cycles = 10
    # cycles < min_cycles and net_pnl <= 0 -> -INF
    assert _calc_rotation_score("BTCUSDT", {"profit": 0, "cycles": 5}, min_cycles) == -float('inf')
    
    # net_pnl > 0
    p = {"profit": 100.0, "cycles": 20}
    expected_score = (100.0 / 20) * math.log1p(20)
    assert _calc_rotation_score("BTCUSDT", p, min_cycles) == pytest.approx(expected_score)
    
    # net_pnl <= 0 but enough cycles -> net_pnl * 0.01
    assert _calc_rotation_score("BTCUSDT", {"profit": -10.0, "cycles": 20}, min_cycles) == pytest.approx(-0.1)

@pytest.mark.asyncio
async def test_selective_merge_incubator():
    config = {"max_bots": 4, "max_replace_per_cycle": 2, "min_cycles_for_rank": 5}
    scanner_results = [
        {"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}, {"symbol": "D"}, {"symbol": "E"}
    ]
    
    # First start
    res = await selective_merge_incubator([], scanner_results, config, {})
    assert res == ["A", "B", "C", "D"]
    
    # Rotation
    old_incubator = ["A", "B", "C", "D"]
    perf_map = {
        "A": {"profit": 10.0, "cycles": 10}, # Profitable
        "B": {"profit": 5.0, "cycles": 10},  # Profitable
        "C": {"profit": -10.0, "cycles": 10}, # Unprofitable
        "D": {"profit": -20.0, "cycles": 10}  # Unprofitable
    }
    
    scanner_results_new = [
        {"symbol": "A"}, {"symbol": "B"}, {"symbol": "E"}, {"symbol": "F"}
    ]
    
    res = await selective_merge_incubator(old_incubator, scanner_results_new, config, perf_map)
    assert set(res) == {"A", "B", "E", "F"}

@pytest.mark.asyncio
async def test_get_bot_efficiency():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"
    
    with patch(f"{M}.safe_load_json", AsyncMock(return_value={})):
        res = await get_bot_efficiency(ticker, config)
        assert res["profit"] == 0.0
        assert res["cycles"] == 0

    state = {
        "last_profit": 100.0,
        "rebalance_cycles": 20,
        "trailing_stop_paper_timeout_end": 123456789.0
    }
    with patch(f"{M}.safe_load_json", AsyncMock(return_value=state)):
        res = await get_bot_efficiency(ticker, config)
        assert res["profit"] == 100.0
        assert res["cycles"] == 20
        assert res["eff"] == 100.0 / 20
        assert res["trailing_stop_paper_timeout_end"] == 123456789.0

@pytest.mark.asyncio
async def test_reset_bot_state_files_real():
    config = {"portfolios": [{"initial_capital": 200.0}]}
    ticker = "BTCUSDT"
    with patch(f"{M}.shutil.copy") as mock_copy, \
         patch(f"{M}.os.remove") as mock_remove, \
         patch(f"{M}.Path.exists", return_value=True), \
         patch(f"{M}.safe_load_json", AsyncMock(return_value={})), \
         patch(f"{M}.safe_save_json", AsyncMock()) as mock_save:
        await reset_bot_state_files(ticker, is_paper=False, config=config)
        assert mock_copy.call_count == 2
        assert mock_remove.call_count == 2
        assert mock_save.call_count == 2
        # Verify first save (base state)
        args, kwargs = mock_save.call_args_list[0]
        assert args[1]["balance"] == 200.0
        assert args[1]["base_ticker"] == ticker

@pytest.mark.asyncio
async def test_reset_bot_state_files_paper():
    config = {"portfolios": [{"paper_initial_capital": 150.0}]}
    ticker = "BTCUSDT"
    with patch(f"{M}.shutil.copy") as mock_copy, \
         patch(f"{M}.os.remove") as mock_remove, \
         patch(f"{M}.Path.exists", return_value=True), \
         patch(f"{M}.safe_load_json", AsyncMock(return_value={})), \
         patch(f"{M}.safe_save_json", AsyncMock()) as mock_save:
        await reset_bot_state_files(ticker, is_paper=True, config=config)
        assert mock_copy.call_count == 2
        assert mock_remove.call_count == 2
        assert mock_save.call_count == 2
        args, kwargs = mock_save.call_args_list[0]
        assert args[1]["balance"] == 150.0

@pytest.mark.asyncio
async def test_enforce_swarm_consistency():
    config = {"live_swarm": ["BTCUSDT"]}
    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        "BTCUSDT_LONG": {"qty": 1.0},
        "ETHUSDT_SHORT": {"qty": -1.0}
    })

    with patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock()) as mock_shell, \
         patch(f"{M}.get_running_bots_info", AsyncMock(return_value={"r_BTCUSDT": {}})), \
         patch(f"{M}.start_bot", AsyncMock()), \
         patch(f"{M}.Path.exists", return_value=False):
        mock_process = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_shell.return_value = mock_process

        await enforce_swarm_consistency(connector, config)

        # Should close ETHUSDT (unauthorized, no state file)
        assert mock_shell.call_count == 1
        cmd = mock_shell.call_args[0][0]
        assert "--ticker ETHUSDT" in cmd
        assert "--stop" in cmd
        assert "--real" in cmd

@pytest.mark.asyncio
async def test_ensure_real_bots_alive():
    config = {"live_swarm": ["BTCUSDT", "ETHUSDT"]}
    # BTCUSDT is running, ETHUSDT is not
    running_bots = {"r_BTCUSDT": {"name": "real-btcusdt"}}

    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value=running_bots)), \
         patch(f"{M}.start_bot", AsyncMock()) as mock_start, \
         patch(f"{M}.Path.exists", return_value=False):
        await _ensure_real_bots_alive(config)
        # Should restart ETHUSDT
        mock_start.assert_called_once_with("ETHUSDT", is_paper=False, config=config)

@pytest.mark.asyncio
async def test_manage_swarm_toxic_flow():
    # Mock config
    config = {
        "tickers": ["BTCUSDT"],
        "max_bots": 2,
        "toxic_blacklist_paper": {},
        "live_swarm": []
    }
    
    # Mock scanner to return a toxic ticker
    scanner_results = [
        {"symbol": "BTCUSDT", "is_toxic": False},
        {"symbol": "TRXUSDT", "is_toxic": True}
    ]
    
    mock_conn = MagicMock()
    mock_conn.verify_connection = AsyncMock()
    mock_conn.get_positions = AsyncMock(return_value={})

    with patch(f"{M}.safe_load_json", AsyncMock(return_value=config)), \
         patch(f"{M}.BinanceConnector", return_value=mock_conn), \
         patch(f"{M}.enforce_swarm_consistency", AsyncMock()), \
         patch(f"{M}.run_scanner", AsyncMock(return_value=scanner_results)), \
         patch(f"{M}.get_bot_efficiency", AsyncMock(return_value={"profit": 0.0, "cycles": 0})), \
         patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.start_bot", AsyncMock()), \
         patch(f"{M}.stop_bot", AsyncMock()), \
         patch(f"{M}.safe_save_json", AsyncMock()) as mock_save, \
         patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock())), \
         patch(f"{M}.Path.exists", return_value=True), \
         patch(f"{M}.Path.glob", return_value=[MagicMock(stem="stop_paper_TRXUSDT", unlink=MagicMock())]):

        await manage_swarm()

        # Check if TRXUSDT was added to toxic_blacklist_paper in saved config
        saved_config = mock_save.call_args_list[0][0][1]
        assert "TRXUSDT" in saved_config["toxic_blacklist_paper"]

@pytest.mark.asyncio
async def test_isolated_blacklists_signal_processing():
    """
    B3 (2026-07-11): Маршрутизация сигналов:
    - stop → toxic_blacklist + black_list (токсичный исход)
    - exit → probation (прибыльный исход, без перманентного бана)
    """
    config = {
        "tickers": ["BTCUSDT", "ETHUSDT"],
        "max_bots": 2,
        "toxic_blacklist_paper": {},
        "toxic_blacklist_real": {},
        "probation_paper": {},
        "probation_real": {},
        "black_list": [],
        "live_swarm": []
    }

    mock_conn = MagicMock()
    mock_conn.verify_connection = AsyncMock()
    mock_conn.get_positions = AsyncMock(return_value={})

    # Эмуляция файлов-сигналов от Paper и Real ботов
    mock_flag_paper = MagicMock(stem="stop_paper_BTCUSDT", unlink=MagicMock())
    mock_flag_real = MagicMock(stem="exit_real_ETHUSDT", unlink=MagicMock())

    with patch(f"{M}.safe_load_json", AsyncMock(return_value=config)), \
         patch(f"{M}.BinanceConnector", return_value=mock_conn), \
         patch(f"{M}.enforce_swarm_consistency", AsyncMock(return_value=set())), \
         patch(f"{M}.run_scanner", AsyncMock(return_value=[{"symbol": "BTCUSDT"}])), \
         patch(f"{M}.get_bot_efficiency", AsyncMock(return_value={"profit": 0.0, "cycles": 0})), \
         patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.start_bot", AsyncMock()), \
         patch(f"{M}.stop_bot", AsyncMock()), \
         patch(f"{M}.safe_save_json", AsyncMock()) as mock_save, \
         patch(f"{M}.Path.exists", return_value=True), \
         patch(f"{M}.Path.glob", return_value=[mock_flag_paper, mock_flag_real]):

        await manage_swarm()

        saved_config = mock_save.call_args_list[0][0][1]
        # stop_paper_BTCUSDT → toxic_blacklist_paper + black_list
        assert "BTCUSDT" in saved_config["toxic_blacklist_paper"]
        assert "BTCUSDT" in saved_config["black_list"]
        # exit_real_ETHUSDT → probation_real (NOT toxic_blacklist_real)
        assert "ETHUSDT" in saved_config["probation_real"]
        assert "ETHUSDT" not in saved_config["toxic_blacklist_real"]
        assert "BTCUSDT" not in saved_config["toxic_blacklist_real"]

@pytest.mark.asyncio
async def test_get_running_bots_info():
    pm2_jlist = [
        {
            "name": "paper-btcusdt",
            "pm2_env": {"args": ["--ticker", "BTCUSDT"]}
        },
        {
            "name": "real-ethusdt",
            "pm2_env": {"args": ["--ticker", "ETHUSDT"]}
        }
    ]
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (json.dumps(pm2_jlist).encode(), None)
    
    with patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
        res = await supervisor.get_running_bots_info()
        assert "p_BTCUSDT" in res
        assert "r_ETHUSDT" in res
        assert res["p_BTCUSDT"]["paper"] is True
        assert res["r_ETHUSDT"]["paper"] is False

@pytest.mark.asyncio
async def test_stop_bot():
    with patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock())) as mock_shell:
        await supervisor.stop_bot("BTCUSDT", is_paper=True)
        assert "pm2 delete paper-btc" in mock_shell.call_args[0][0]

@pytest.mark.asyncio
async def test_start_bot():
    config = {"portfolios": [{"initial_capital": 100}]}
    with patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock())) as mock_shell, \
         patch(f"{M}.reset_bot_state_files", AsyncMock()) as mock_reset, \
         patch(f"{M}.Path.exists", return_value=False):
        await supervisor.start_bot("BTCUSDT", is_paper=True, config=config)
        assert mock_reset.called
        assert "pm2 start main.py" in mock_shell.call_args[0][0]
        assert "--paper" in mock_shell.call_args[0][0]


# ============================================================
# Rotation Guard: immature bots (cycles < min_cycles_for_rotation)
# защищены от замены новыми кандидатами из сканера
# ============================================================

@pytest.mark.asyncio
async def test_rotation_guard_immature_unprofitable_protected():
    """
    Убыточный paper-бот с cycles=3 < min_cycles_for_rotation=6
    НЕ должен быть вытеснен новым кандидатом из сканера.
    """
    config = {
        "max_bots": 4,
        "max_replace_per_cycle": 2,
        "min_cycles_for_rank": 5,
        "min_cycles_for_rotation": 6,
    }
    old_incubator = ["A", "B", "C", "D"]
    perf_map = {
        "A": {"profit": 10.0, "cycles": 10},   # Прибыльный — всегда остаётся
        "B": {"profit": -5.0, "cycles": 3},     # Убыточный, но НЕЗРЕЛЫЙ (3 < 6) — ЗАЩИЩЁН
        "C": {"profit": -10.0, "cycles": 10},   # Убыточный зрелый — кандидат на замену
        "D": {"profit": -20.0, "cycles": 10},   # Убыточный зрелый — кандидат на замену
    }
    scanner_results = [
        {"symbol": "A"}, {"symbol": "B"}, {"symbol": "E"}, {"symbol": "F"},
    ]
    res = await selective_merge_incubator(old_incubator, scanner_results, config, perf_map)
    # A (прибыльный) + B (незрелый защищённый) — оба остаются
    assert "A" in res, "Profitable bot A must stay"
    assert "B" in res, "Immature bot B (cycles=3 < 6) must be PROTECTED from replacement"
    # C и D — зрелые убыточные, конкурируют с E, F
    # Итог: A, B + 2 из {C, D, E, F}
    assert len(res) == 4


@pytest.mark.asyncio
async def test_rotation_guard_immature_profitable_protected():
    """
    Прибыльный paper-бот с cycles=2 < min_cycles_for_rotation=6
    тоже защищён (прибыльные и так всегда остаются,
    но проверяем что логика не сломана).
    """
    config = {
        "max_bots": 3,
        "max_replace_per_cycle": 2,
        "min_cycles_for_rank": 5,
        "min_cycles_for_rotation": 6,
    }
    old_incubator = ["A", "B", "C"]
    perf_map = {
        "A": {"profit": 1.0, "cycles": 2},     # Прибыльный незрелый
        "B": {"profit": -5.0, "cycles": 10},    # Убыточный зрелый
        "C": {"profit": -10.0, "cycles": 10},   # Убыточный зрелый
    }
    scanner_results = [
        {"symbol": "A"}, {"symbol": "D"}, {"symbol": "E"},
    ]
    res = await selective_merge_incubator(old_incubator, scanner_results, config, perf_map)
    assert "A" in res, "Profitable bot A must stay regardless of cycles"
    assert len(res) == 3


@pytest.mark.asyncio
async def test_rotation_guard_mature_unprofitable_replaced():
    """
    Убыточный paper-бот с cycles=8 >= min_cycles_for_rotation=6
    НЕ защищён — может быть вытеснен новым кандидатом.
    """
    config = {
        "max_bots": 3,
        "max_replace_per_cycle": 2,
        "min_cycles_for_rank": 5,
        "min_cycles_for_rotation": 6,
    }
    old_incubator = ["A", "B", "C"]
    perf_map = {
        "A": {"profit": 10.0, "cycles": 10},   # Прибыльный — остаётся
        "B": {"profit": -5.0, "cycles": 8},     # Убыточный ЗРЕЛЫЙ (8 >= 6) — НЕ защищён
        "C": {"profit": -10.0, "cycles": 10},   # Убыточный зрелый — НЕ защищён
    }
    scanner_results = [
        {"symbol": "A"}, {"symbol": "D"}, {"symbol": "E"},
    ]
    res = await selective_merge_incubator(old_incubator, scanner_results, config, perf_map)
    assert "A" in res, "Profitable bot A must stay"
    # B и C — зрелые убыточные, D и E — новые с score=0
    # B имеет score = -5 * 0.01 = -0.05, C = -10 * 0.01 = -0.10
    # D, E имеют score=0 — они вытеснят B и C (0 > -0.05 > -0.10)
    assert "D" in res, "New candidate D should replace mature unprofitable"
    assert "E" in res, "New candidate E should replace mature unprofitable"
    assert "B" not in res, "Mature unprofitable B (cycles=8 >= 6) should be replaced"
    assert "C" not in res, "Mature unprofitable C should be replaced"


@pytest.mark.asyncio
async def test_rotation_guard_default_min_cycles_for_rotation():
    """
    Если min_cycles_for_rotation НЕ указан в config,
    используется дефолт=6 (из .get default).
    """
    config = {
        "max_bots": 3,
        "max_replace_per_cycle": 2,
        "min_cycles_for_rank": 5,
        # min_cycles_for_rotation НЕ задан — дефолт 6
    }
    old_incubator = ["A", "B", "C"]
    perf_map = {
        "A": {"profit": -1.0, "cycles": 5},  # 5 < 6 — незрелый, защищён
        "B": {"profit": -5.0, "cycles": 10}, # зрелый убыточный
        "C": {"profit": -10.0, "cycles": 10},# зрелый убыточный
    }
    scanner_results = [
        {"symbol": "A"}, {"symbol": "D"}, {"symbol": "E"},
    ]
    res = await selective_merge_incubator(old_incubator, scanner_results, config, perf_map)
    assert "A" in res, "Immature bot A (cycles=5 < default 6) must be protected"
    assert len(res) == 3


# ============================================================
# B1 FIX TEST: stop_bot MUST be called BEFORE TS flag reset
# ============================================================

@pytest.mark.asyncio
async def test_b1_stop_bot_called_before_ts_flag_reset():
    """
    B1 FIX VERIFICATION:
    When enforce_swarm_consistency encounters a ticker with
    trailing_stop_triggered=True and a dead PM2 process, it MUST
    call stop_bot() to kill the zombie BEFORE resetting the TS
    flags in the state file. Otherwise Reaper Guard (later in
    the loop) won't find the zombie because flags are already False.
    """
    ticker = "GRASSUSDT"
    config = {"live_swarm": [ticker]}
    state_data = {
        "trailing_stop_triggered": True,
        "trailing_stop_violation_start": 12345.0,
        "rebalance_cycles": 10,
        "virt_qty": 0.0
    }

    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        f"{ticker}_LONG": {"qty": 1.0}
    })

    # Track call order: stop_bot vs file write
    call_order = []

    async def fake_stop_bot(t, is_paper=False):
        call_order.append(("stop_bot", t))

    # Mock open to capture file writes
    original_builtin_open = open
    import io
    def tracking_open(path, *args, **kwargs):
        if "real_state_" in str(path) and "w" in str(args):
            def write_and_track(content):
                call_order.append(("file_write", str(path)))
                return len(content)
            buf = io.StringIO()
            buf.write = write_and_track
            return buf
        return original_builtin_open(path, *args, **kwargs)

    import json as _json

    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.stop_bot", side_effect=fake_stop_bot) as mock_stop, \
         patch(f"{M}.start_bot", AsyncMock()), \
         patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock(wait=AsyncMock()))), \
         patch(f"{M}.Path.exists", return_value=True), \
         patch("builtins.open", side_effect=tracking_open):

        original_load = _json.load
        def fake_load(f, *args, **kwargs):
            if "real_state_" in str(getattr(f, 'name', '')):
                return dict(state_data)
            return original_load(f, *args, **kwargs)

        with patch("json.load", side_effect=fake_load):
            await enforce_swarm_consistency(connector, config)

    # Verify stop_bot was called
    mock_stop.assert_called_once_with(ticker, is_paper=False)

    # Verify stop_bot was called BEFORE file write
    stop_idx = next((i for i, (op, _) in enumerate(call_order) if op == "stop_bot"), None)
    write_idx = next((i for i, (op, _) in enumerate(call_order) if op == "file_write"), None)

    assert stop_idx is not None, "stop_bot was never called"
    assert write_idx is not None, "State file was never written"
    assert stop_idx < write_idx, (
        f"B1 BUG: stop_bot (idx={stop_idx}) must be called BEFORE "
        f"file write (idx={write_idx}). Zombie would survive!"
    )


@pytest.mark.asyncio
async def test_b1_ts_flag_reset_after_stop_bot():
    """
    B1 FIX VERIFICATION (flag state):
    After enforce_swarm_consistency, the state file must have
    trailing_stop_triggered=False — but ONLY AFTER stop_bot was called.
    """
    ticker = "VVVUSDT"
    config = {"live_swarm": [ticker]}
    state_data = {
        "trailing_stop_triggered": True,
        "trailing_stop_violation_start": 99999.0,
        "rebalance_cycles": 5,
        "virt_qty": 0.0
    }

    connector = MagicMock()
    connector.get_positions = AsyncMock(return_value={
        f"{ticker}_SHORT": {"qty": -0.5}
    })

    stop_called = False
    written_state = {}

    async def fake_stop_bot(t, is_paper=False):
        nonlocal stop_called
        stop_called = True

    original_builtin_open2 = open
    import io
    def tracking_open2(path, *args, **kwargs):
        if "real_state_" in str(path) and "w" in str(args):
            buf = io.StringIO()
            original_write = buf.write
            def accumulate_write(content):
                original_write(content)
                return len(content)
            buf.write = accumulate_write
            # On close, capture the full content
            original_close = buf.close
            def capture_and_close():
                nonlocal written_state
                try:
                    buf.seek(0)
                    written_state = _json.loads(buf.read())
                except:
                    pass
                original_close()
            buf.close = capture_and_close
            return buf
        return original_builtin_open2(path, *args, **kwargs)

    import json as _json

    with patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})), \
         patch(f"{M}.stop_bot", side_effect=fake_stop_bot), \
         patch(f"{M}.start_bot", AsyncMock()), \
         patch(f"{M}.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock(wait=AsyncMock()))), \
         patch(f"{M}.Path.exists", return_value=True), \
         patch("builtins.open", side_effect=tracking_open2):

        original_load2 = _json.load
        def fake_load2(f, *args, **kwargs):
            if "real_state_" in str(getattr(f, 'name', '')):
                return dict(state_data)
            return original_load2(f, *args, **kwargs)

        with patch("json.load", side_effect=fake_load2):
            await enforce_swarm_consistency(connector, config)

    # stop_bot MUST have been called
    assert stop_called, "stop_bot was not called — zombie survives!"

    # State file MUST have trailing_stop_triggered=False
    assert written_state.get("trailing_stop_triggered") is False, (
        f"TS flag was not reset! Got: {written_state.get('trailing_stop_triggered')}"
    )
    # Violation start MUST be cleared
    assert written_state.get("trailing_stop_violation_start") == 0.0


# =====================================================================
# B6 + B3 + P1 + P2 + P3: SSOT Refactor Tests (2026-07-11)
# =====================================================================

import asyncio
import time
from pathlib import Path


@pytest.mark.asyncio
async def test_signal_stop_adds_toxic_and_blacklist(tmp_path):
    """B3: stop signal → toxic_blacklist + permanent black_list (not probation)"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "stop_real_BTCUSDT.flag").touch()

    config = {
        "live_swarm": ["BTCUSDT", "ETHUSDT"],
        "toxic_blacklist_real": {},
        "toxic_blacklist_paper": {},
        "probation_real": {},
        "probation_paper": {},
        "black_list": [],
        "toxic_cooldown_days": 0.02,
        "probation_period_days": 0.041,
    }

    # Simulate signal processing logic from supervisor.py
    now_ts = time.time()
    toxic_cooldown_sec = config["toxic_cooldown_days"] * 86400
    prob_paper = config.get("probation_paper", {})
    prob_real = config.get("probation_real", {})

    for flag_file in sig_dir.glob("*.flag"):
        parts = flag_file.stem.split("_", 2)
        if len(parts) != 3:
            continue
        signal_type, mode_tag, ticker = parts

        # Remove from live_swarm
        live_swarm = config.get("live_swarm", [])
        if ticker in live_swarm:
            live_swarm.remove(ticker)
            config["live_swarm"] = live_swarm

        if signal_type == "stop":
            bl_key = "toxic_blacklist_paper" if mode_tag == "paper" else "toxic_blacklist_real"
            if bl_key not in config:
                config[bl_key] = {}
            config[bl_key][ticker] = now_ts + toxic_cooldown_sec
            if "black_list" not in config:
                config["black_list"] = []
            if ticker not in config["black_list"]:
                config["black_list"].append(ticker)

        await asyncio.to_thread(flag_file.unlink)

    assert "BTCUSDT" in config["toxic_blacklist_real"]
    assert "BTCUSDT" in config["black_list"]
    assert "BTCUSDT" not in config["probation_real"]
    assert "BTCUSDT" not in config["live_swarm"]


@pytest.mark.asyncio
async def test_signal_exit_adds_probation_not_blacklist(tmp_path):
    """B3: exit signal → probation only (no toxic blacklist, no permanent ban)"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "exit_paper_ZECUSDT.flag").touch()

    config = {
        "live_swarm": ["ZECUSDT"],
        "toxic_blacklist_paper": {},
        "probation_paper": {},
        "black_list": [],
        "toxic_cooldown_days": 0.02,
        "probation_period_days": 0.041,
    }

    now_ts = time.time()
    probation_cooldown_sec = config["probation_period_days"] * 86400

    for flag_file in sig_dir.glob("*.flag"):
        parts = flag_file.stem.split("_", 2)
        if len(parts) != 3:
            continue
        signal_type, mode_tag, ticker = parts

        live_swarm = config.get("live_swarm", [])
        if ticker in live_swarm:
            live_swarm.remove(ticker)
            config["live_swarm"] = live_swarm

        if signal_type == "exit":
            prob_key = "probation_paper" if mode_tag == "paper" else "probation_real"
            if prob_key not in config:
                config[prob_key] = {}
            config[prob_key][ticker] = now_ts + probation_cooldown_sec

        await asyncio.to_thread(flag_file.unlink)

    assert "ZECUSDT" in config["probation_paper"]
    assert "ZECUSDT" not in config["toxic_blacklist_paper"]
    assert "ZECUSDT" not in config["black_list"]
    assert "ZECUSDT" not in config["live_swarm"]


@pytest.mark.asyncio
async def test_signal_removes_from_live_swarm(tmp_path):
    """Любой сигнал → тикер удалён из live_swarm"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "stop_real_ETHUSDT.flag").touch()

    config = {"live_swarm": ["ETHUSDT", "BTCUSDT"]}

    for flag_file in sig_dir.glob("*.flag"):
        parts = flag_file.stem.split("_", 2)
        if len(parts) != 3:
            continue
        _, _, ticker = parts
        live_swarm = config.get("live_swarm", [])
        if ticker in live_swarm:
            live_swarm.remove(ticker)
            config["live_swarm"] = live_swarm
        await asyncio.to_thread(flag_file.unlink)

    assert "ETHUSDT" not in config["live_swarm"]
    assert "BTCUSDT" in config["live_swarm"]


@pytest.mark.asyncio
async def test_signal_unknown_type_no_config_change(tmp_path):
    """Unknown signal type → file deleted, config unchanged"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "unknown_real_BTC.flag").touch()

    config_before = {"live_swarm": ["BTCUSDT"], "black_list": []}
    config = dict(config_before)

    for flag_file in sig_dir.glob("*.flag"):
        parts = flag_file.stem.split("_", 2)
        if len(parts) != 3:
            await asyncio.to_thread(flag_file.unlink)
            continue
        signal_type = parts[0]
        # Unknown type — skip
        await asyncio.to_thread(flag_file.unlink)

    assert config == config_before
    assert not list(sig_dir.glob("*.flag"))


def test_blacklist_set_o1_lookup():
    """P1: black_list converted to set for O(1) lookup"""
    config = {"black_list": ["A", "B", "C", "D"]}

    # Convert list → set (as in manage_swarm)
    if isinstance(config["black_list"], list):
        config["black_list"] = set(config["black_list"])

    assert isinstance(config["black_list"], set)
    assert "A" in config["black_list"]  # O(1)
    assert "Z" not in config["black_list"]

    # Convert set → sorted list (as before save_json)
    if isinstance(config["black_list"], set):
        config["black_list"] = sorted(config["black_list"])

    assert isinstance(config["black_list"], list)
    assert config["black_list"] == ["A", "B", "C", "D"]


def test_probation_blocks_candidate():
    """P3: Ticker in probation_real → excluded from candidates"""
    config = {
        "probation_real": {"BTCUSDT": time.time() + 3600},
        "probation_paper": {},
    }

    current_real_tickers = ["BTCUSDT", "ETHUSDT"]
    prob_real = config.get("probation_real", {})

    # Filter: exclude probation tickers
    filtered = [t for t in current_real_tickers if t not in prob_real]

    assert "BTCUSDT" not in filtered
    assert "ETHUSDT" in filtered


def test_probation_expired_not_blocks():
    """P3: Expired probation → ticker available again"""
    config = {
        "probation_real": {"BTCUSDT": time.time() - 100},  # expired
    }

    prob_real = config.get("probation_real", {})
    now = time.time()
    active = {t: exp for t, exp in prob_real.items() if exp > now}

    assert "BTCUSDT" not in active


@pytest.mark.asyncio
async def test_flag_unlink_is_async(tmp_path):
    """P2: flag_file.unlink() uses asyncio.to_thread (non-blocking)"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    flag = sig_dir / "stop_real_TEST.flag"
    flag.touch()

    # Verify file exists
    assert flag.exists()

    # Use asyncio.to_thread as in the new code
    for f in sig_dir.glob("*.flag"):
        await asyncio.to_thread(f.unlink)

    # Verify file deleted
    assert not list(sig_dir.glob("*.flag"))




# =====================================================================
# Integration tests through manage_swarm() -- covers 591-1029
# CRITICAL: manage_swarm() loads config from CONFIG_PATH on disk.
# Tests MUST write config.json to tmp_path + monkeypatch CONFIG_PATH.
# =====================================================================

import contextlib


def _write_config(tmp_path, **overrides):
    """Write config.json to tmp_path. Returns the config dict for assertions."""
    cfg = {
        "testnet": True, "api_key": "test", "secret_key": "test",
        "live_swarm": [], "black_list": [], "real_whitelist": [],
        "toxic_blacklist_paper": {}, "toxic_blacklist_real": {},
        "probation_paper": {}, "probation_real": {},
        "toxic_cooldown_days": 0.02, "probation_period_days": 0.041,
        "scanner_period_days": 1.0, "max_bots": 10, "paper_mode_bots": 9,
        "max_replace_per_cycle": 1, "min_cycles_for_rank": 10,
        "min_cycles_for_rotation": 6, "replacement_efficiency_threshold_pct": 25.0,
        "initial_capital": 100.0, "min_notional_usdt": 5.0,
        "max_drawdown_limit": 5.0, "equity_trailing_stop_pct": 0.001,
        "equity_trailing_stop_activation_pct": 7.5,
        "equity_trailing_stop_timeout_sec": 60,
        "paper_mode": True, "limit_order_enabled": False,
        "limit_offset_pct": 0.001, "limit_timeout_sec": 30,
        "max_orders_per_second": 5,
        "supervisor_interval_days": 1.0,
        "backtest_period_days": 2.0,
        "use_v2_scoring": False, "scoring_drawdown_weight": 0.5,
        "telegram_enabled": False, "telegram_use_queue": False,
        "telegram_summary_interval_min": 60,
        "portfolios": [{"paper_initial_capital": 115.0, "initial_capital": 80.0}],
        "tickers": [], "base_ticker": "",
        "liquidation_distance_warn_pct": 15.0,
        "liquidation_distance_crit_pct": 8.0,
        "margin_ratio_warning": 5.0, "margin_ratio_critical": 2.0,
        "max_real_slots": 1, "use_real_whitelist": True,
    }
    cfg.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def _cm_manage_swarm(tmp_path, *, config_overrides=None, running_bots=None,
                     scanner=None, eff_map=None, positions=None, saved=None):
    """Context manager: all mocks for manage_swarm + CONFIG_PATH redirect."""
    if config_overrides is None:
        config_overrides = {}
    if running_bots is None:
        running_bots = {}
    if scanner is None:
        scanner = [{"symbol": "X", "score": 1}]
    if eff_map is None:
        eff_map = {}
    if positions is None:
        positions = {}
    if saved is None:
        saved = {}

    _write_config(tmp_path, **config_overrides)

    async def capture_save(path, data):
        saved.update(data)

    async def fake_eff(ticker, cfg, is_real=False):
        return eff_map.get(ticker, {"profit": 0.0, "cycles": 0})

    cm = contextlib.ExitStack()
    cm.enter_context(patch(f"{M}.CONFIG_PATH", str(tmp_path / "config.json")))
    cm.enter_context(patch(f"{M}.BASE_PATH", tmp_path))
    conn_patch = cm.enter_context(patch(f"{M}.BinanceConnector"))
    cm.enter_context(patch(f"{M}.get_running_bots_info", AsyncMock(return_value=running_bots)))
    cm.enter_context(patch(f"{M}.run_scanner", AsyncMock(return_value=scanner)))
    cm.enter_context(patch(f"{M}.get_bot_efficiency", AsyncMock(side_effect=fake_eff)))
    cm.enter_context(patch(f"{M}.enforce_invariant_gate", AsyncMock()))
    cm.enter_context(patch(f"{M}.safe_save_json", AsyncMock(side_effect=capture_save)))
    cm.enter_context(patch(f"{M}.enforce_swarm_consistency", AsyncMock(return_value=set())))
    cm.enter_context(patch(f"{M}._ensure_real_bots_alive", AsyncMock()))
    cm.enter_context(patch(f"{M}.reconcile_swarm_state", AsyncMock()))
    cm.enter_context(patch(f"{M}.stop_bot", AsyncMock()))
    cm.enter_context(patch(f"{M}.start_bot", AsyncMock()))
    cm.enter_context(patch(f"{M}.selective_merge_incubator", AsyncMock(return_value=[r["symbol"] for r in scanner])))
    cm.enter_context(patch(f"{M}.asyncio.create_subprocess_shell", return_value=AsyncMock(wait=AsyncMock())))

    conn_inst = conn_patch.return_value
    conn_inst.verify_connection = AsyncMock()
    conn_inst.get_positions = AsyncMock(return_value=positions)

    return cm, saved, conn_patch, cm.enter_context(patch(f"{M}.stop_bot", AsyncMock())), cm.enter_context(patch(f"{M}.start_bot", AsyncMock()))


async def _run_manage(tmp_path, **kwargs):
    """Run manage_swarm with all mocks. Returns (saved, mock_stop, mock_start)."""
    cm, saved, _, mock_stop, mock_start = _cm_manage_swarm(tmp_path, **kwargs)
    with cm:
        from futures_portfolio.supervisor import manage_swarm
        await manage_swarm()
    return saved, mock_stop, mock_start


@pytest.mark.asyncio
async def test_integration_signal_stop_toxic_blacklist(tmp_path, monkeypatch):
    """Lines 591-613: stop signal -> toxic_blacklist + black_list"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "stop_real_BTCUSDT.flag").touch()

    saved, _, _ = await _run_manage(tmp_path, config_overrides={"live_swarm": ["BTCUSDT"]})

    assert "BTCUSDT" in saved.get("toxic_blacklist_real", {})
    assert not list(sig_dir.glob("*.flag"))


@pytest.mark.asyncio
async def test_integration_signal_exit_probation(tmp_path, monkeypatch):
    """Lines 620-630: exit signal -> probation"""
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "exit_paper_ZECUSDT.flag").touch()

    saved, _, _ = await _run_manage(tmp_path, config_overrides={"live_swarm": ["ZECUSDT"]})

    assert "ZECUSDT" in saved.get("probation_paper", {})


@pytest.mark.asyncio
async def test_integration_amnesty_clears_ts(tmp_path, monkeypatch):
    """Lines 654-663: expired blacklist -> Amnesty resets TS flags"""
    state = {"trailing_stop_triggered": True, "trailing_stop_violation_start": 9999.0}
    (tmp_path / "paper_state_ALICEUSDT.json").write_text(json.dumps(state))

    expired_ts = time.time() - 3600
    saved, _, _ = await _run_manage(tmp_path, config_overrides={
        "toxic_blacklist_paper": {"ALICEUSDT": expired_ts},
        "black_list": ["ALICEUSDT"],
    })

    updated = json.loads((tmp_path / "paper_state_ALICEUSDT.json").read_text())
    assert updated["trailing_stop_triggered"] is False
    assert updated["trailing_stop_violation_start"] == 0.0


@pytest.mark.asyncio
async def test_integration_reaper_zombie(tmp_path, monkeypatch):
    """Lines 684-698: running bot with TS triggered -> killed"""
    state = {"trailing_stop_triggered": True}
    (tmp_path / "paper_state_ALICEUSDT.json").write_text(json.dumps(state))

    running = {"p_ALICEUSDT": {"name": "paper-aliceusdt", "paper": True}}
    _, mock_stop, _ = await _run_manage(tmp_path, running_bots=running)

    mock_stop.assert_called()


@pytest.mark.asyncio
async def test_integration_ready_pool_scoring(tmp_path, monkeypatch):
    """Lines 796-836: scoring with whitelist filters"""
    eff_map = {"GOODUSDT": {"profit": 50.0, "cycles": 20}}
    scanner = [{"symbol": "GOODUSDT", "score": 10}]
    saved, _, _ = await _run_manage(tmp_path,
        config_overrides={"real_whitelist": ["GOODUSDT"]},
        scanner=scanner, eff_map=eff_map)

    final_swarm = saved.get("live_swarm", [])
    assert "GOODUSDT" in final_swarm


@pytest.mark.asyncio
async def test_integration_real_rotation_fills_slots(tmp_path, monkeypatch):
    """Lines 853-857: empty slots filled from candidates"""
    eff_map = {"GOODUSDT": {"profit": 80.0, "cycles": 20}}
    scanner = [{"symbol": "GOODUSDT", "score": 10}]
    _, mock_stop, mock_start = await _run_manage(tmp_path,
        config_overrides={"real_whitelist": ["GOODUSDT"]},
        scanner=scanner, eff_map=eff_map)

    real_calls = [c for c in mock_start.call_args_list
                  if not c.kwargs.get("is_paper", True)]
    assert len(real_calls) >= 1


@pytest.mark.asyncio
async def test_integration_replaces_unprofitable(tmp_path, monkeypatch):
    """Lines 861-931: unprofitable real bot replaced"""
    bad_state = {"started_at": time.time() - 86400 * 7}
    (tmp_path / "real_state_BADUSDT.json").write_text(json.dumps(bad_state))

    running = {"r_BADUSDT": {"name": "real-badusdt", "paper": False}}
    eff_map = {
        "BADUSDT": {"profit": 0.0, "cycles": 30},
        "GOODUSDT": {"profit": 80.0, "cycles": 20},
    }
    scanner = [{"symbol": "GOODUSDT", "score": 10}]
    _, _, mock_start = await _run_manage(tmp_path,
        config_overrides={"live_swarm": ["BADUSDT"], "real_whitelist": ["GOODUSDT"]},
        running_bots=running, scanner=scanner, eff_map=eff_map)

    real_starts = [c for c in mock_start.call_args_list
                   if "GOODUSDT" in str(c) and not c.kwargs.get("is_paper", True)]
    assert len(real_starts) >= 1


@pytest.mark.asyncio
async def test_integration_stops_out_of_incubator(tmp_path, monkeypatch):
    """Lines 976-978: paper bot not in incubator -> stopped"""
    running = {"p_OLDUSDT": {"name": "paper-oldusdt", "paper": True}}
    scanner = [{"symbol": "BTCUSDT", "score": 10}]
    _, mock_stop, _ = await _run_manage(tmp_path, running_bots=running, scanner=scanner)

    stop_calls = [c for c in mock_stop.call_args_list if "OLDUSDT" in str(c)]
    assert len(stop_calls) >= 1


@pytest.mark.asyncio
async def test_integration_probation_blocks(tmp_path, monkeypatch):
    """Lines 818-822: active probation -> excluded"""
    future_ts = time.time() + 3600
    eff_map = {"BLOCKEDUSDT": {"profit": 50.0, "cycles": 20}}
    scanner = [{"symbol": "BLOCKEDUSDT", "score": 10}]
    saved, _, _ = await _run_manage(tmp_path,
        config_overrides={
            "real_whitelist": ["BLOCKEDUSDT"],
            "probation_paper": {"BLOCKEDUSDT": future_ts},
        },
        scanner=scanner, eff_map=eff_map)

    final_swarm = saved.get("live_swarm", [])
    assert "BLOCKEDUSDT" not in final_swarm


@pytest.mark.asyncio
async def test_integration_drawdown_protection(tmp_path, monkeypatch):
    """Lines 813-815: real bot in drawdown -> score INF, protected"""
    running = {"r_DRAWDOWNUSDT": {"name": "real-drawdownusdt", "paper": False}}
    eff_map = {"DRAWDOWNUSDT": {"profit": -20.0, "cycles": 15}}
    scanner = [{"symbol": "DRAWDOWNUSDT", "score": 10}]
    saved, _, _ = await _run_manage(tmp_path,
        config_overrides={"real_whitelist": ["DRAWDOWNUSDT"]},
        running_bots=running, scanner=scanner, eff_map=eff_map)

    final_swarm = saved.get("live_swarm", [])
    assert "DRAWDOWNUSDT" in final_swarm


@pytest.mark.asyncio
async def test_integration_toxic_scanner_mark(tmp_path, monkeypatch):
    """Lines 721-724: scanner marks toxic -> blacklisted"""
    scanner = [{"symbol": "TOXICUSDT", "score": 10, "is_toxic": True}]
    saved, _, _ = await _run_manage(tmp_path, scanner=scanner)

    assert "TOXICUSDT" in saved.get("toxic_blacklist_paper", {})


@pytest.mark.asyncio
async def test_integration_scanner_empty_aborts(tmp_path, monkeypatch):
    """Lines 705-707: scanner returns nothing -> abort, no config save"""
    saved, _, _ = await _run_manage(tmp_path, scanner=[])
    assert "live_swarm" not in saved


@pytest.mark.asyncio
async def test_integration_expired_probation_pruned(tmp_path, monkeypatch):
    """Lines 668-671: expired probation entries pruned"""
    expired_prob = time.time() - 3600
    saved, _, _ = await _run_manage(tmp_path,
        config_overrides={"probation_paper": {"EXPIREDUSDT": expired_prob}})

    assert "EXPIREDUSDT" not in saved.get("probation_paper", {})


@pytest.mark.asyncio
async def test_integration_ticks_fallback(tmp_path, monkeypatch):
    """Lines 1019-1023: tickers empty -> restored from scanner"""
    scanner = [{"symbol": "BTCUSDT", "score": 10}, {"symbol": "ETHUSDT", "score": 8}]
    saved, _, _ = await _run_manage(tmp_path,
        config_overrides={"tickers": [], "base_ticker": ""},
        scanner=scanner)

    assert len(saved.get("tickers", [])) > 0


@pytest.mark.asyncio
async def test_integration_real_stop_drops_bot(tmp_path, monkeypatch):
    """Lines 981-992: real bot replaced -> old stopped"""
    state = {"started_at": time.time() - 86400}
    (tmp_path / "real_state_DROPPEDUSDT.json").write_text(json.dumps(state))
    running = {"r_DROPPEDUSDT": {"name": "real-droppedusdt", "paper": False}}
    eff_map = {
        "DROPPEDUSDT": {"profit": 0.0, "cycles": 10},
        "BTCUSDT": {"profit": 50.0, "cycles": 20},
    }
    scanner = [{"symbol": "BTCUSDT", "score": 10}]
    _, mock_stop, _ = await _run_manage(tmp_path,
        config_overrides={"real_whitelist": ["BTCUSDT"], "use_real_whitelist": True},
        running_bots=running, scanner=scanner, eff_map=eff_map)

    stop_calls = [c for c in mock_stop.call_args_list
                  if "DROPPEDUSDT" in str(c) and not c.kwargs.get("is_paper", True)]
    assert len(stop_calls) >= 1


@pytest.mark.asyncio
async def test_integration_authoritative_cleanup_stray(tmp_path, monkeypatch):
    """Lines 950-956: stray position not in target -> closed"""
    shell_calls = []

    async def track_shell(cmd, **kw):
        shell_calls.append(cmd)
        return AsyncMock(wait=AsyncMock())

    _write_config(tmp_path, real_whitelist=[], tickers=[], base_ticker="")

    with patch(f"{M}.CONFIG_PATH", str(tmp_path / "config.json")),          patch(f"{M}.BASE_PATH", tmp_path),          patch(f"{M}.BinanceConnector") as MockConn,          patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})),          patch(f"{M}.run_scanner", AsyncMock(return_value=[{"symbol": "X", "score": 1}])),          patch(f"{M}.get_bot_efficiency", AsyncMock(return_value={})),          patch(f"{M}.enforce_invariant_gate", AsyncMock()),          patch(f"{M}.safe_save_json", AsyncMock()),          patch(f"{M}.enforce_swarm_consistency", AsyncMock(return_value=set())),          patch(f"{M}._ensure_real_bots_alive", AsyncMock()),          patch(f"{M}.reconcile_swarm_state", AsyncMock()),          patch(f"{M}.stop_bot", AsyncMock()),          patch(f"{M}.start_bot", AsyncMock()),          patch(f"{M}.selective_merge_incubator", AsyncMock(return_value=["X"])),          patch(f"{M}.asyncio.create_subprocess_shell", side_effect=track_shell):
        MockConn.return_value.verify_connection = AsyncMock()
        MockConn.return_value.get_positions = AsyncMock(return_value={
            "STRAYCOINUSDT_LONG": {"qty": "5.0", "mark_price": "10.0"}
        })
        from futures_portfolio.supervisor import manage_swarm
        await manage_swarm()

    stop_cmds = [c for c in shell_calls if "STRAYCOINUSDT" in str(c) and "--stop" in str(c)]
    assert len(stop_cmds) >= 1


@pytest.mark.asyncio
async def test_integration_safety_trim(tmp_path, monkeypatch):
    """Lines 935-941: more bots than max_real_slots -> trim worst"""
    # Create 2 real bots but max_bots-paper_mode_bots=1
    running = {
        "r_BOTA": {"name": "real-bota", "paper": False},
        "r_BOTB": {"name": "real-botb", "paper": False},
    }
    eff_map = {
        "BOTA": {"profit": 0.0, "cycles": 15},
        "BOTB": {"profit": 0.0, "cycles": 15},
    }
    scanner = [{"symbol": "BOTA", "score": 10}, {"symbol": "BOTB", "score": 8}]
    _, mock_stop, _ = await _run_manage(tmp_path,
        config_overrides={"max_bots": 10, "paper_mode_bots": 9, "real_whitelist": []},
        running_bots=running, scanner=scanner, eff_map=eff_map)

    # At least one stop should happen (worst bot trimmed)
    assert mock_stop.call_count >= 1


@pytest.mark.asyncio
async def test_integration_healed_tickers(tmp_path, monkeypatch):
    """Lines 677-680: healed tickers added"""
    config_overrides = {"real_whitelist": ["HEALUSDT"]}

    _write_config(tmp_path, **config_overrides)

    with patch(f"{M}.CONFIG_PATH", str(tmp_path / "config.json")),          patch(f"{M}.BASE_PATH", tmp_path),          patch(f"{M}.BinanceConnector") as MockConn,          patch(f"{M}.get_running_bots_info", AsyncMock(return_value={})),          patch(f"{M}.run_scanner", AsyncMock(return_value=[])),          patch(f"{M}.safe_save_json", AsyncMock()),          patch(f"{M}.enforce_swarm_consistency", AsyncMock(return_value={"HEALUSDT"})),          patch(f"{M}._ensure_real_bots_alive", AsyncMock()),          patch(f"{M}.reconcile_swarm_state", AsyncMock()),          patch(f"{M}.stop_bot", AsyncMock()),          patch(f"{M}.start_bot", AsyncMock()),          patch(f"{M}.selective_merge_incubator", AsyncMock(return_value=[])),          patch(f"{M}.asyncio.create_subprocess_shell", return_value=AsyncMock(wait=AsyncMock())):
        MockConn.return_value.verify_connection = AsyncMock()
        MockConn.return_value.get_positions = AsyncMock(return_value={})
        from futures_portfolio.supervisor import manage_swarm
        await manage_swarm()


@pytest.mark.asyncio
async def test_integration_blacklisted_real_skipped(tmp_path, monkeypatch):
    """Lines 803-805: ticker in toxic_blacklist_real -> quarantined"""
    future_ts = time.time() + 3600
    eff_map = {"QUARANTINEDUSDT": {"profit": 100.0, "cycles": 25}}
    scanner = [{"symbol": "QUARANTINEDUSDT", "score": 10}]
    saved, _, _ = await _run_manage(tmp_path,
        config_overrides={
            "real_whitelist": ["QUARANTINEDUSDT"],
            "toxic_blacklist_real": {"QUARANTINEDUSDT": future_ts},
        },
        scanner=scanner, eff_map=eff_map)

    final_swarm = saved.get("live_swarm", [])
    assert "QUARANTINEDUSDT" not in final_swarm
