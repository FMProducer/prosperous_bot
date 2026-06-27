import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from typing import List, Dict, Any

import sys
from pathlib import Path

# Добавляем путь к futures_portfolio, чтобы импортировать supervisor
sys.path.append(str(Path(__file__).resolve().parent.parent / "futures_portfolio"))

from supervisor import reconcile_swarm_state

@pytest.fixture
def mock_pm2_processes() -> List[Dict[str, Any]]:
    # Naming convention: f"{mode}-{ticker.replace('USDT', '').lower()}"
    # BTCUSDT -> btc, ETHUSDT -> eth, SOLUSDT -> sol, DOGEUSDT -> doge
    return [
        {"name": "paper-btc", "pm2_env": {"status": "online"}},
        {"name": "paper-eth", "pm2_env": {"status": "online"}},
        {"name": "real-sol", "pm2_env": {"status": "online"}},
        {"name": "paper-doge", "pm2_env": {"status": "stopped"}}, # Orphaned phantom
    ]

@pytest.mark.asyncio
@patch("supervisor.get_pm2_processes")
@patch("supervisor.asyncio.create_subprocess_exec")
async def test_reconcile_swarm_state_kills_orphans(
    mock_create_subprocess_exec,
    mock_get_pm2_processes,
    mock_pm2_processes
):
    # Setup
    mock_get_pm2_processes.return_value = mock_pm2_processes

    mock_proc = AsyncMock()
    mock_proc.wait = AsyncMock(return_value=0)
    mock_create_subprocess_exec.return_value = mock_proc

    # Execute: Define only btc and eth as valid paper targets. doge should be killed.
    target_tickers = ["BTCUSDT", "ETHUSDT"]
    await reconcile_swarm_state(target_tickers, mode="paper")

    # Assert
    # Should call pm2 delete for "paper-doge"
    mock_create_subprocess_exec.assert_any_call(
        "pm2", "delete", "paper-doge",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )

    # Should also call pm2 save once
    mock_create_subprocess_exec.assert_any_call(
        "pm2", "save",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )

    # Total calls: 1 for delete, 1 for save
    assert mock_create_subprocess_exec.call_count == 2

@pytest.mark.asyncio
@patch("supervisor.get_pm2_processes")
@patch("supervisor.asyncio.create_subprocess_exec")
async def test_reconcile_swarm_state_perfect_sync(mock_exec, mock_get_pm2_procs):
    mock_get_pm2_procs.return_value = [{"name": "real-btc", "pm2_env": {"status": "online"}}]
    await reconcile_swarm_state(["BTCUSDT"], mode="real")
    # No orphans to kill, no pm2 save should be called
    mock_exec.assert_not_called()
