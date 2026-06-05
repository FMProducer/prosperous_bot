import pytest
import math
import json
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
import supervisor
from supervisor import calculate_bot_score, _calc_rotation_score, selective_merge_incubator, get_bot_efficiency, reset_bot_state_files, enforce_swarm_consistency, _ensure_real_bots_alive, manage_swarm

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
    # scanner_top will be ["A", "B", "C", "D"]
    # new tickers from scanner? wait scanner_results has A, B, C, D, E.
    # scanner_top = ["A", "B", "C", "D"]
    # scanner_new = [] because all are in old_set
    # Wait, if scanner_top is same as old, no new tickers.
    
    scanner_results_new = [
        {"symbol": "A"}, {"symbol": "B"}, {"symbol": "E"}, {"symbol": "F"}
    ]
    # scanner_top = ["A", "B", "E", "F"]
    # profitable = {A, B}
    # unprofitable = {C, D}
    # scanner_new = {E, F}
    # competitors: C, D (scored) + E, F (score 0)
    # C score = -10 * 0.01 = -0.1
    # D score = -20 * 0.01 = -0.2
    # E, F score = 0
    # competitors sorted: E(0), F(0), C(-0.1), D(-0.2)
    # final = [A, B] + [E, F] = [A, B, E, F] (if max_bots=4)
    
    res = await selective_merge_incubator(old_incubator, scanner_results_new, config, perf_map)
    assert set(res) == {"A", "B", "E", "F"}

@pytest.mark.asyncio
async def test_get_bot_efficiency():
    config = {"min_cycles_for_rank": 10}
    ticker = "BTCUSDT"
    
    with patch("supervisor.safe_load_json", AsyncMock(return_value={})) as mock_load:
        res = await get_bot_efficiency(ticker, config)
        assert res["profit"] == 0.0
        assert res["cycles"] == 0
        
    state = {
        "last_profit": 100.0,
        "rebalance_cycles": 20,
        "trailing_stop_paper_timeout_end": 123456789.0
    }
    with patch("supervisor.safe_load_json", AsyncMock(return_value=state)):
        res = await get_bot_efficiency(ticker, config)
        assert res["profit"] == 100.0
        assert res["cycles"] == 20
        assert res["eff"] == 100.0 / 20
        assert res["trailing_stop_paper_timeout_end"] == 123456789.0

@pytest.mark.asyncio
async def test_reset_bot_state_files_real():
    config = {"portfolios": [{"initial_capital": 200.0}]}
    ticker = "BTCUSDT"
    with patch("supervisor.shutil.copy") as mock_copy, \
         patch("supervisor.os.remove") as mock_remove, \
         patch("supervisor.Path.exists", return_value=True), \
         patch("supervisor.safe_load_json", AsyncMock(return_value={})), \
         patch("supervisor.safe_save_json", AsyncMock()) as mock_save:
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
    with patch("supervisor.shutil.copy") as mock_copy, \
         patch("supervisor.os.remove") as mock_remove, \
         patch("supervisor.Path.exists", return_value=True), \
         patch("supervisor.safe_load_json", AsyncMock(return_value={})), \
         patch("supervisor.safe_save_json", AsyncMock()) as mock_save:
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
    # Mock active positions: BTCUSDT (authorized) and ETHUSDT (unauthorized)
    connector.get_positions = AsyncMock(return_value={
        "BTCUSDT_LONG": {"qty": 1.0},
        "ETHUSDT_SHORT": {"qty": -1.0}
    })
    
    with patch("supervisor.asyncio.create_subprocess_shell", AsyncMock()) as mock_shell, \
         patch("supervisor.get_running_bots_info", AsyncMock(return_value={"r_BTCUSDT": {}})):
        mock_process = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_shell.return_value = mock_process
        
        await enforce_swarm_consistency(connector, config)
        
        # Should close ETHUSDT (get_running_bots_info is mocked, so only 1 shell call for closing)
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
    
    with patch("supervisor.get_running_bots_info", AsyncMock(return_value=running_bots)), \
         patch("supervisor.start_bot", AsyncMock()) as mock_start:
        await _ensure_real_bots_alive(config)
        # Should restart ETHUSDT
        mock_start.assert_called_once_with("ETHUSDT", is_paper=False, config=config)

@pytest.mark.asyncio
async def test_manage_swarm_toxic_flow():
    # Mock config
    config = {
        "tickers": ["BTCUSDT"],
        "max_bots": 2,
        "toxic_blacklist": {},
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

    with patch("supervisor.safe_load_json", AsyncMock(side_effect=[config, {}])), \
         patch("supervisor.BinanceConnector", return_value=mock_conn), \
         patch("supervisor.enforce_swarm_consistency", AsyncMock()), \
         patch("supervisor.run_scanner", AsyncMock(return_value=scanner_results)), \
         patch("supervisor.get_bot_efficiency", AsyncMock(return_value={"profit": 0.0, "cycles": 0})), \
         patch("supervisor.get_running_bots_info", AsyncMock(return_value={})), \
         patch("supervisor.start_bot", AsyncMock()), \
         patch("supervisor.stop_bot", AsyncMock()), \
         patch("supervisor.safe_save_json", AsyncMock()) as mock_save, \
         patch("supervisor.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock())), \
         patch("supervisor.Path.exists", return_value=True), \
         patch("supervisor.Path.glob", return_value=[MagicMock(stem="stop_ETHUSDT", unlink=MagicMock())]):
        
        await manage_swarm()
        
        # Check if TRXUSDT was added to toxic_blacklist in saved config
        saved_config = mock_save.call_args_list[0][0][1]
        assert "TRXUSDT" in saved_config["toxic_blacklist"]
        # Check if ETHUSDT (from stop signal) was added
        assert "ETHUSDT" in saved_config["toxic_blacklist"]

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
    
    with patch("supervisor.asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
        res = await supervisor.get_running_bots_info()
        assert "p_BTCUSDT" in res
        assert "r_ETHUSDT" in res
        assert res["p_BTCUSDT"]["paper"] is True
        assert res["r_ETHUSDT"]["paper"] is False

@pytest.mark.asyncio
async def test_stop_bot():
    with patch("supervisor.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock())) as mock_shell:
        await supervisor.stop_bot("BTCUSDT", is_paper=True)
        assert "pm2 delete paper-btc" in mock_shell.call_args[0][0]

@pytest.mark.asyncio
async def test_start_bot():
    config = {"portfolios": [{"initial_capital": 100}]}
    with patch("supervisor.asyncio.create_subprocess_shell", AsyncMock(return_value=AsyncMock())) as mock_shell, \
         patch("supervisor.reset_bot_state_files", AsyncMock()) as mock_reset:
        await supervisor.start_bot("BTCUSDT", is_paper=True, config=config)
        assert mock_reset.called
        assert "pm2 start main.py" in mock_shell.call_args[0][0]
        assert "--paper" in mock_shell.call_args[0][0]
