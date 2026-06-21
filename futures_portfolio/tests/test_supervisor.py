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
    Однозначно подтверждает функциональность парсинга сигналов супервайзером
    и их маршрутизацию в строго изолированные списки (toxic_blacklist_real и toxic_blacklist_paper).
    """
    config = {
        "tickers": ["BTCUSDT", "ETHUSDT"],
        "max_bots": 2,
        "toxic_blacklist_paper": {},
        "toxic_blacklist_real": {},
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
        assert "BTCUSDT" in saved_config["toxic_blacklist_paper"]
        assert "ETHUSDT" not in saved_config["toxic_blacklist_paper"]
        assert "ETHUSDT" in saved_config["toxic_blacklist_real"]
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
