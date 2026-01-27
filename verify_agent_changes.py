import sys
import os
import numpy as np
import torch
import logging

# Add the directory to sys.path
sys.path.append('third_party/rl-trading-binance')

from agent import D3QN_PER_Agent

# Mock DuelingQNetwork to avoid complex initialization
class MockPolicyNet:
    def __init__(self, input_shape, additional_feats):
        self.input_shape = input_shape
        self.additional_feats = additional_feats

    def eval(self):
        pass

def test_agent_flattening():
    logging.basicConfig(level=logging.DEBUG)

    state_shape = (3, 10, 1)
    action_dim = 3
    additional_feats = 5

    # Initialize agent
    # We need to mock policy_net before it's used in _check_flat_state_size
    agent = D3QN_PER_Agent(
        state_shape=state_shape,
        action_dim=action_dim,
        cnn_maps=[16],
        cnn_kernels=[3],
        cnn_strides=[1],
        dense_val=[32],
        dense_adv=[32],
        additional_feats=additional_feats,
        dropout_model=0.1,
        device="cpu",
        gamma=0.99,
        learning_rate=1e-4,
        batch_size=32,
        buffer_size=1000,
        target_update_freq=100,
        train_start=100,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=1000,
        eps_start=1.0,
        eps_end=0.1,
        eps_frames=1000,
        epsilon=0.1,
        max_gradient_norm=1.0
    )

    # Test _flatten_single_state
    state_2d = np.random.rand(30 + additional_feats)
    flat = agent._flatten_single_state(state_2d)
    assert flat.shape == (35,)
    assert flat.dtype == np.float32

    state_3d = np.random.rand(3, 10, 1)
    # The agent expects a flat vector that includes additional_feats.
    # If state_3d doesn't include them, it will fail _check_flat_state_size later.
    # But _flatten_single_state just flattens.
    flat = agent._flatten_single_state(state_3d)
    assert flat.shape == (30,)

    # Test _check_flat_state_size
    # Expected size = 3 * 10 + 5 = 35
    agent._check_flat_state_size(np.random.rand(35))

    try:
        agent._check_flat_state_size(np.random.rand(30))
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        print(f"Caught expected error: {e}")

    # Test _flatten_batch_states
    states_batch = np.random.rand(10, 3, 10, 1)
    flat_batch = agent._flatten_batch_states(states_batch)
    assert flat_batch.shape == (10, 30)

    print("Agent flattening tests passed!")

if __name__ == "__main__":
    test_agent_flattening()
