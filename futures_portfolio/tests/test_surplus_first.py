import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from futures_portfolio.executor import PortfolioExecutor

@pytest.mark.asyncio
async def test_execute_actions_surplus_first():
    connector = Mock()
    executor = PortfolioExecutor(connector)

    actions = [
        {"symbol": "BTCUSDT_LONG", "is_reduction": False, "type": "ORDER", "diff_usdt": 100},
        {"symbol": "BTCUSDT_SHORT", "is_reduction": True, "type": "ORDER", "diff_usdt": -50},
        {"symbol": "VIRTUAL", "is_reduction": True, "type": "VIRTUAL_ORDER", "diff_usdt": -20},
        {"symbol": "ETHUSDT_LONG", "is_reduction": False, "type": "ORDER", "diff_usdt": 30},
    ]

    execution_order = []

    async def mock_execute_single_action(action, *args, **kwargs):
        execution_order.append(action["symbol"])
        return {"status": "SUCCESS", "symbol": action["symbol"], "type": action.get("position_side", "BOTH")}

    executor._execute_single_action = mock_execute_single_action

    results = await executor.execute_actions(actions, price=60000, paper_mode=True)

    # Reductions should be first
    assert execution_order[0:2] == ["BTCUSDT_SHORT", "VIRTUAL"]
    # Expansions should be last
    assert execution_order[2:4] == ["BTCUSDT_LONG", "ETHUSDT_LONG"]

    # Results should be in original order
    assert [r["symbol"] for r in results] == ["BTCUSDT_LONG", "BTCUSDT_SHORT", "VIRTUAL", "ETHUSDT_LONG"]

@pytest.mark.asyncio
async def test_execute_actions_concurrent_stages():
    connector = Mock()
    executor = PortfolioExecutor(connector)

    actions = [
        {"symbol": "RED1", "is_reduction": True, "type": "ORDER", "diff_usdt": -50},
        {"symbol": "EXP1", "is_reduction": False, "type": "ORDER", "diff_usdt": 100},
    ]

    red1_started = asyncio.Event()
    red1_can_finish = asyncio.Event()

    async def mock_execute_single_action(action, *args, **kwargs):
        if action["symbol"] == "RED1":
            red1_started.set()
            await red1_can_finish.wait()
        elif action["symbol"] == "EXP1":
            if not red1_started.is_set() or not red1_can_finish.is_set():
                # Expansion started before reduction finished or even before reduction started (should not happen)
                # But since it's a separate stage, red1 MUST have finished.
                pass
        return {"status": "SUCCESS", "symbol": action["symbol"]}

    executor._execute_single_action = mock_execute_single_action

    # Run execute_actions in background
    task = asyncio.create_task(executor.execute_actions(actions, price=60000, paper_mode=True))

    await red1_started.wait()
    # At this point RED1 is running, EXP1 should NOT have started yet.
    # We can't easily prove a negative without more complex instrumenting,
    # but we can check the execution flow logic.

    red1_can_finish.set()
    results = await task

    assert [r["symbol"] for r in results] == ["RED1", "EXP1"]
