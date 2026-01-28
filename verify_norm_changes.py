
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), "third_party/rl-trading-binance"))

from trading_environment_z import TradingEnvironment

def test_env_normalization():
    # Mock data: 100 steps, 5 channels (OHLCV)
    # Volume (index 4) has some large values
    data = np.random.rand(100, 5).astype(np.float32)
    data[:, 4] = np.array([10, 100, 1000] * 33 + [10000]).astype(np.float32)

    datachannels = ["open", "high", "low", "close", "volume"]
    pricechannels = ["open", "high", "low", "close"]
    volumechannels = ["volume"]
    otherchannels = []

    # Test case 1: Static Normalization
    norm_stats = {
        "TEST": {
            "mean": [0.5, 0.5, 0.5, 0.5, np.log1p(1000.0)],
            "std": [0.1, 0.1, 0.1, 0.1, 1.0]
        }
    }

    env = TradingEnvironment(
        sequences=[data],
        keys=["TEST_20230101"],
        render_mode=None,
        full_seq_len=100,
        num_features=5,
        num_actions=3,
        flat_state_size=50,
        initial_balance=1000,
        pre_signal_len=10,
        datachannels=datachannels,
        slippage=0.001,
        transaction_fee=0.001,
        agent_session_len=50,
        agent_history_len=10,
        input_history_len=10,
        pricechannels=pricechannels,
        volumechannels=volumechannels,
        otherchannels=otherchannels,
        action_history_len=0,
        inaction_penalty_ratio=0.0,
        norm_stats=norm_stats
    )

    obs, _ = env.reset(options={"forced_index": 0})
    # Observation contains normalized history.
    # For index 4, it should be (log1p(volume) - mean) / std
    # The last element of the history part of obs (first 10*5 = 50 elements)
    # for channel 4 is at index (10-1)*5 + 4 = 49

    expected_log_vol = np.log1p(data[9, 4])
    expected_norm_vol = (expected_log_vol - norm_stats["TEST"]["mean"][4]) / (norm_stats["TEST"]["std"][4] + 1e-8)
    expected_norm_vol = np.clip(expected_norm_vol, -5.0, 5.0)

    actual_norm_vol = obs[49]

    print(f"Static Norm - Expected log vol: {expected_log_vol:.4f}")
    print(f"Static Norm - Expected norm vol: {expected_norm_vol:.4f}")
    print(f"Static Norm - Actual norm vol: {actual_norm_vol:.4f}")

    assert np.allclose(actual_norm_vol, expected_norm_vol, atol=1e-5)
    print("Static Normalization Test Passed!")

    # Test case 2: Rolling Normalization
    env_rolling = TradingEnvironment(
        sequences=[data],
        keys=["TEST_20230101"],
        render_mode=None,
        full_seq_len=100,
        num_features=5,
        num_actions=3,
        flat_state_size=50,
        initial_balance=1000,
        pre_signal_len=10,
        datachannels=datachannels,
        slippage=0.001,
        transaction_fee=0.001,
        agent_session_len=50,
        agent_history_len=10,
        input_history_len=10,
        pricechannels=pricechannels,
        volumechannels=volumechannels,
        otherchannels=otherchannels,
        action_history_len=0,
        inaction_penalty_ratio=0.0,
        norm_stats=None,
        use_rolling_norm=True
    )

    obs_r, _ = env_rolling.reset(options={"forced_index": 0})
    # Rolling norm for volume: (log1p(vol) - mean(log1p(v_window))) / std(log1p(v_window))
    v_window = data[0:10, 4]
    log_v_window = np.log1p(v_window)
    v_mean = np.mean(log_v_window)
    v_std = np.std(log_v_window) + 1e-8

    expected_r_norm_vol = (np.log1p(data[9, 4]) - v_mean) / v_std
    expected_r_norm_vol = np.clip(expected_r_norm_vol, -5.0, 5.0)

    actual_r_norm_vol = obs_r[49]

    print(f"Rolling Norm - Expected norm vol: {expected_r_norm_vol:.4f}")
    print(f"Rolling Norm - Actual norm vol: {actual_r_norm_vol:.4f}")

    assert np.allclose(actual_r_norm_vol, expected_r_norm_vol, atol=1e-5)
    print("Rolling Normalization Test Passed!")

    # Test case 3: Pre-normalized data (skip norm)
    # We pre-normalize data manually
    means = np.array([0.5, 0.5, 0.5, 0.5, 5.0])
    stds = np.array([0.1, 0.1, 0.1, 0.1, 1.0])
    data_pre = data.copy()
    data_pre[:, 4] = np.log1p(data_pre[:, 4])
    data_pre = (data_pre - means) / (stds + 1e-8)
    data_pre = np.clip(data_pre, -5.0, 5.0)

    env_pre = TradingEnvironment(
        sequences=[data_pre],
        keys=["TEST_20230101"],
        render_mode=None,
        full_seq_len=100,
        num_features=5,
        num_actions=3,
        flat_state_size=50,
        initial_balance=1000,
        pre_signal_len=10,
        datachannels=datachannels,
        slippage=0.001,
        transaction_fee=0.001,
        agent_session_len=50,
        agent_history_len=10,
        input_history_len=10,
        pricechannels=pricechannels,
        volumechannels=volumechannels,
        otherchannels=otherchannels,
        action_history_len=0,
        inaction_penalty_ratio=0.0,
        norm_stats=None,
        use_rolling_norm=False
    )

    obs_p, _ = env_pre.reset(options={"forced_index": 0})
    # Should be exactly the same as data_pre[0:10]
    expected_p_vol = data_pre[9, 4]
    actual_p_vol = obs_p[49]

    print(f"Pre-norm - Expected vol: {expected_p_vol:.4f}")
    print(f"Pre-norm - Actual vol: {actual_p_vol:.4f}")

    assert np.allclose(actual_p_vol, expected_p_vol, atol=1e-5)
    print("Pre-normalization Test Passed!")

if __name__ == "__main__":
    try:
        test_env_normalization()
        print("All Environment Normalization Tests Passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
