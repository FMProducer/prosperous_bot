
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from futures_portfolio.supervisor.swarm_manager import manage_swarm

@pytest.mark.asyncio
async def test_manage_swarm_respects_whitelist_during_rotation():
    # Setup mock config
    config = {
        "api_key": "fake",
        "secret_key": "fake",
        "testnet": True,
        "live_swarm": ["BTCUSDT"],
        "real_whitelist": ["ETHUSDT"],
        "tickers": ["BTCUSDT", "ETHUSDT"],
        "max_bots": 10,
        "paper_mode_bots": 8, # max_real_slots = 10 - 8 = 2
        "min_cycles_for_rank": 10
    }

    # Mock dependencies
    with patch("futures_portfolio.supervisor.swarm_manager.safe_load_json", AsyncMock(return_value=config)), \
         patch("futures_portfolio.supervisor.swarm_manager.safe_save_json", AsyncMock()), \
         patch("futures_portfolio.supervisor.swarm_manager.BinanceConnector") as MockConnector, \
         patch("futures_portfolio.supervisor.swarm_manager.run_scanner", AsyncMock(return_value=[{"symbol": "BTCUSDT", "score": 100}])), \
         patch("futures_portfolio.supervisor.swarm_manager.get_running_bots_info", AsyncMock(return_value={
             "r_BTCUSDT": {"name": "real-btc", "paper": False},
             "r_ETHUSDT": {"name": "real-eth", "paper": False}
         })), \
         patch("futures_portfolio.supervisor.swarm_manager.get_bot_efficiency", AsyncMock(return_value={"profit": 10.0, "cycles": 20})), \
         patch("futures_portfolio.supervisor.swarm_manager.stop_bot", AsyncMock()) as mock_stop_bot, \
         patch("futures_portfolio.supervisor.swarm_manager.start_bot", AsyncMock()), \
         patch("asyncio.create_subprocess_shell", AsyncMock()) as mock_shell, \
         patch("futures_portfolio.supervisor.swarm_manager.enforce_swarm_consistency", AsyncMock()):

        mock_connector = MockConnector.return_value
        mock_connector.verify_connection = AsyncMock()
        mock_connector.get_positions = AsyncMock(return_value={})

        # We want to see if ETHUSDT is stopped.
        # target_real_bots will likely only include BTCUSDT (if it's the top scorer)
        # ETHUSDT is in running_bots but might not be in target_real_bots

        await manage_swarm()

        # Check if stop_bot was called for ETHUSDT
        # In our case, ETHUSDT IS in real_whitelist, so it should NOT be stopped.

        stop_bot_calls = [call.args[0] for call in mock_stop_bot.call_args_list if not call.args[1]] # is_paper=False
        assert "ETHUSDT" not in stop_bot_calls

        # Also check Authoritative Cleanup
        # If we mock positions to include ETHUSDT
        mock_connector.get_positions = AsyncMock(return_value={"ETHUSDT_LONG": {"qty": 1.0}})
        await manage_swarm()

        # Authoritative cleanup uses create_subprocess_shell with --stop
        shell_calls = [call.args[0] for call in mock_shell.call_args_list]
        assert not any("--ticker ETHUSDT" in c and "--stop" in c for c in shell_calls)
