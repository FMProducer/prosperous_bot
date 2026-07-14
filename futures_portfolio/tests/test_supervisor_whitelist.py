
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from futures_portfolio.supervisor.swarm_manager import enforce_swarm_consistency

@pytest.mark.asyncio
async def test_enforce_swarm_consistency_respects_whitelist():
    # Setup mock connector
    connector = AsyncMock()
    connector.get_positions.return_value = {
        "BTCUSDT_LONG": {"qty": 1.0},
        "ETHUSDT_SHORT": {"qty": -2.0},
        "SOLUSDT_LONG": {"qty": 5.0}
    }

    # Setup config: BTC in live_swarm, ETH in real_whitelist, SOL in neither
    config = {
        "live_swarm": ["BTCUSDT"],
        "real_whitelist": ["ETHUSDT"]
    }

    # We want to check if it tries to close SOLUSDT but NOT BTCUSDT or ETHUSDT
    with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell, \
         patch("futures_portfolio.supervisor.swarm_manager.get_running_bots_info", AsyncMock(return_value={"r_BTCUSDT": {}, "r_ETHUSDT": {}})):
        mock_process = AsyncMock()
        mock_shell.return_value.wait = AsyncMock(return_value=0)
        mock_shell.return_value = mock_process

        await enforce_swarm_consistency(connector, config)

        # Check calls to create_subprocess_shell
        # It should be called for SOLUSDT
        calls = [call.args[0] for call in mock_shell.call_args_list]

        assert any("--ticker SOLUSDT" in c for c in calls)
        assert not any("--ticker BTCUSDT" in c for c in calls)
        assert not any("--ticker ETHUSDT" in c for c in calls)

@pytest.mark.asyncio
async def test_enforce_swarm_consistency_all_allowed():
    connector = AsyncMock()
    connector.get_positions.return_value = {
        "BTCUSDT_LONG": {"qty": 1.0},
        "ETHUSDT_SHORT": {"qty": -2.0}
    }

    config = {
        "live_swarm": ["BTCUSDT"],
        "real_whitelist": ["ETHUSDT"]
    }

    with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_shell, \
         patch("futures_portfolio.supervisor.swarm_manager.get_running_bots_info", AsyncMock(return_value={"r_BTCUSDT": {}, "r_ETHUSDT": {}})):
        await enforce_swarm_consistency(connector, config)

        assert mock_shell.call_count == 0
