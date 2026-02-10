import pytest
import sys
import os
from pathlib import Path

# Ensure we can import from project root and third_party
project_root = os.getcwd()
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'third_party/rl-trading-binance'))

from trading_environment import TradingEnvironment

class MockEnv(TradingEnvironment):
    def __init__(self, agent_mode):
        self.agent_mode = agent_mode
        # Mock minimal attributes to avoid full init
        self.stats = {}
        self.sequences = []
        self.num_actions = 3 if agent_mode == "UNIVERSAL" else 2

@pytest.mark.parametrize("mode, action_in, expected_direction", [
    ("UNIVERSAL", 0, 0), ("UNIVERSAL", 1, 1), ("UNIVERSAL", 2, -1),
    ("LONG_ONLY", 0, 0), ("LONG_ONLY", 1, 1),
    ("SHORT_ONLY", 0, 0), ("SHORT_ONLY", 1, -1),
    ("MIRROR_SHORT", 0, 0), ("MIRROR_SHORT", 1, 1),
])
def test_get_direction_logic(mode, action_in, expected_direction):
    env = MockEnv(agent_mode=mode)
    assert env._get_direction(action_in) == expected_direction

@pytest.mark.parametrize("mode, action_in, internal_code", [
    ("SHORT_ONLY", 1, 2), # Key Fix: Action 1 maps to Internal 2 (Short)
    ("LONG_ONLY", 1, 1),  # Action 1 maps to Internal 1 (Long)
])
def test_step_mapping_integration(mode, action_in, internal_code):
    env = MockEnv(agent_mode=mode)
    direction = env._get_direction(action_in)

    # Simulate step() logic
    mapped_action = 0
    if direction == 1: mapped_action = 1
    elif direction == -1: mapped_action = 2

    assert mapped_action == internal_code
