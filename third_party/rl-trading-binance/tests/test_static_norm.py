
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_environment_z import TradingEnvironment

def test_static_normalization():
    # Setup data
    raw_seq = np.array([
        [100, 105, 95, 102, 1000],
        [102, 107, 101, 104, 1100],
        [104, 109, 103, 106, 1200],
    ], dtype=np.float32)

    norm_stats = {
        'TEST': {
            'mean': [102.0, 107.0, 99.0, 104.0, 1100.0],
            'std': [2.0, 2.0, 2.0, 2.0, 100.0]
        }
    }

    env_kwargs = {
        "sequences": [raw_seq],
        "keys": ['TEST_ASSET'],
        "render_mode": None,
        "full_seq_len": 3,
        "num_features": 5,
        "num_actions": 3,
        "flat_state_size": 15,
        "initial_balance": 10000.0,
        "pre_signal_len": 1,
        "datachannels": ["open", "high", "low", "close", "volume"],
        "slippage": 0.0,
        "transaction_fee": 0.0,
        "agent_session_len": 2,
        "agent_history_len": 1,
        "input_history_len": 1,
        "pricechannels": ["open", "high", "low", "close"],
        "volumechannels": ["volume"],
        "otherchannels": [],
        "action_history_len": 0,
        "inaction_penalty_ratio": 0.0,
        "norm_stats": norm_stats
    }

    env = TradingEnvironment(**env_kwargs)
    env.reset()

    # current_asset_name is TEST
    # current_step is 0
    # _get_observation should return the first row normalized by norm_stats['TEST']
    obs = env._get_observation()

    # If using MLP mode, obs is a flat vector where the first num_features elements are the normalized data
    expected_norm = (raw_seq[0] - np.array(norm_stats['TEST']['mean'])) / np.array(norm_stats['TEST']['std'])

    # raw_seq[0] = [100, 105, 95, 102, 1000]
    # mean = [102, 107, 99, 104, 1100]
    # std = [2, 2, 2, 2, 100]
    # expected = [-1, -1, -2, -1, -1]

    np.testing.assert_allclose(obs[:5], expected_norm, atol=1e-5)
    print("Static normalization test passed!")

if __name__ == "__main__":
    test_static_normalization()
