
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_environment_z import TradingEnvironment

@pytest.fixture
def base_env_kwargs():
    """Provides a base set of arguments for TradingEnvironment initialization."""
    sequences = [np.random.rand(100, 5)]
    keys = ['TEST']
    return {
        "sequences": sequences,
        "keys": keys,
        "render_mode": None,
        "full_seq_len": 100,
        "num_features": 5,
        "num_actions": 3,
        "flat_state_size": 250,
        "initial_balance": 10000.0,
        "pre_signal_len": 50,
        "datachannels": ["open", "high", "low", "close", "volume"],
        "slippage": 0.001,
        "transaction_fee": 0.001,
        "agent_session_len": 50,
        "agent_history_len": 10,
        "input_history_len": 10,
        "pricechannels": ["open", "high", "low", "close"],
        "volumechannels": ["volume"],
        "otherchannels": [],
        "action_history_len": 0,
        "inaction_penalty_ratio": 0.0,
    }

def test_environment_construction(base_env_kwargs):
    """Tests that the TradingEnvironment can be constructed with different filter_direction settings."""
    # Test with no filter
    env_none = TradingEnvironment(**base_env_kwargs, filter_direction=None)
    assert env_none is not None

    # Test with LONG filter
    env_long = TradingEnvironment(**base_env_kwargs, filter_direction='LONG')
    assert env_long is not None

    # Test with SHORT filter
    env_short = TradingEnvironment(**base_env_kwargs, filter_direction='SHORT')
    assert env_short is not None

def test_short_mode_inversion(base_env_kwargs):
    """Tests the geometric inversion of data in SHORT mode."""
    # Original data for one sequence (falling trend to pass the filter in SHORT mode)
    original_sequence = np.array([
        [110, 115, 95, 105, 1200],  # o, h, l, c, v
        [105, 110, 90, 100, 1000],
    ], dtype=np.float32)

    kwargs = base_env_kwargs.copy()
    kwargs['sequences'] = [original_sequence]
    kwargs['keys'] = ['TEST_ASSET_0']
    kwargs['full_seq_len'] = 2

    # Create the environment in SHORT mode
    env_short = TradingEnvironment(**kwargs, filter_direction='SHORT')

    # Check sequence data inversion
    inverted_sequence = env_short.sequences[0]

    # Expected: open, close, and other price channels are negated
    # Expected: high_short = -low_long, low_short = -high_long
    # Expected: volume is unchanged
    expected_inverted_sequence = np.array([
        [-110, -95, -115, -105, 1200], # -o, -l, -h, -c, v
        [-105, -90, -110, -100, 1000],
    ], dtype=np.float32)

    np.testing.assert_allclose(inverted_sequence, expected_inverted_sequence, rtol=1e-5)
