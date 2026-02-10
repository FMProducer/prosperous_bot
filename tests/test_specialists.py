import pytest
import numpy as np
import sys
import os

# Add the project root and third_party to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'third_party/rl-trading-binance'))

from trading_environment import TradingEnvironment

def mock_env(mode="UNIVERSAL", invert=False, trend='up', use_rm=True):
    # Create dummy OHLCV data
    if trend == 'up':
        prices = np.linspace(100, 150, 100)
    else:
        prices = np.linspace(100, 50, 100)

    data = np.zeros((100, 5))
    data[:, 0] = prices # open
    data[:, 1] = prices + 1 # high
    data[:, 2] = prices - 1 # low
    data[:, 3] = prices # close
    data[:, 4] = 1000 # volume

    sequences = [data]
    keys = ["BTCUSDT_1h"]

    datachannels = ['open', 'high', 'low', 'close', 'volume']
    # Use zero mean and unit std to keep real_price == norm_price (approx)
    stats = {
        "BTCUSDT": {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0, 1.0]
        }
    }

    env = TradingEnvironment(
        sequences=sequences,
        stats=stats,
        keys=keys,
        render_mode=None,
        full_seq_len=100,
        num_features=5,
        num_actions=4, # 0: hold, 1: long, 2: short, 3: close
        flat_state_size=0,
        initial_balance=1000.0,
        pre_signal_len=10,
        datachannels=datachannels,
        slippage=0.0,
        transaction_fee=0.0,
        agent_session_len=80,
        agent_history_len=10,
        input_history_len=10,
        pricechannels=['open', 'high', 'low', 'close'],
        volumechannels=['volume'],
        otherchannels=[],
        action_history_len=5,
        inaction_penalty_ratio=0.0,
        use_risk_management=use_rm,
        filter_direction=mode,
        invert_data=invert,
        stop_loss=0.05,
        take_profit=0.10,
        trailing_stop=0.02,
        trailing_stop_min=0.01,
        delta_p_hysteresis=0.001,
        holding_penalty_multiplier=0.0, # Disable penalties for pure PnL tests
        holding_penalty_threshold=100
    )
    return env

def test_long_specialist_pnl():
    env = mock_env(mode="LONG", trend='up')
    env.reset(options={"forced_index": 0})

    # Step 0: Open Long
    env.step(1)

    # Step 1: Hold, price rises
    _, reward, _, _, info = env.step(0)
    # Since reward is only realized PnL, it's 0 while holding.
    # Check portfolio_value instead.
    assert info['portfolio_value'] > 1000.0, "Long should profit when price rises"

def test_short_specialist_normal_chart_pnl():
    env = mock_env(mode="SHORT", invert=False, trend='down')
    env.reset(options={"forced_index": 0})

    # Step 0: Open Short
    env.step(2)

    # Step 1: Hold, price falls
    _, reward, _, _, info = env.step(0)
    assert info['portfolio_value'] > 1000.0, "Short on normal chart should profit when price falls"

def test_short_specialist_inverted_chart_pnl():
    env = mock_env(mode="SHORT", invert=True, trend='down')
    env.reset(options={"forced_index": 0})

    # Step 0: Open Long (mirrored Short)
    env.step(1)

    # Step 1: Hold, "inverted price" rises (real price falls)
    _, reward, _, _, info = env.step(0)
    assert info['portfolio_value'] > 1000.0, "Short on inverted chart should profit when 'inverted price' rises"

def test_short_tsl_logic():
    env = mock_env(mode="SHORT", invert=False, trend='down', use_rm=True)
    env.trailing_stop = 0.10 # Large trailing stop to avoid instant trigger
    env.reset(options={"forced_index": 0})

    # Step 0: Open Short
    env.step(2)

    # Step 1: Hold. Price falls, TSL should be initialized/updated
    env.step(0)

    assert env.position == -1, "Position should still be open"
    initial_tsl = env.tsl_price
    assert initial_tsl is not None
    assert initial_tsl > env.current_seq[env.pre_signal_len + env.step_idx, env.close_idx]

    # Steps: Price falls. TSL should move DOWN.
    last_tsl = initial_tsl
    for _ in range(10):
        env.step(0)
        current_tsl = env.tsl_price
        assert current_tsl <= last_tsl, "Short TSL must move DOWN or stay same"
        last_tsl = current_tsl

    assert last_tsl < initial_tsl, "Short TSL must have moved DOWN"

def test_short_sl_logic():
    env = mock_env(mode="SHORT", invert=False, trend='up', use_rm=True)
    env.stop_loss = 0.05
    env.reset(options={"forced_index": 0})

    env.step(2)
    assert env.position == -1

    for _ in range(50):
        _, _, terminated, _, info = env.step(0)
        if env.position == 0:
            break

    assert env.position == 0, "Short position should be closed by SL"
    assert "RM_EXIT" in str(info) or env.position == 0

def test_short_tp_logic():
    env = mock_env(mode="SHORT", invert=False, trend='down', use_rm=True)
    env.take_profit = 0.05
    env.trailing_stop = 0.50
    env.reset(options={"forced_index": 0})

    env.step(2)
    assert env.position == -1

    for _ in range(50):
        _, _, terminated, _, info = env.step(0)
        if env.position == 0:
            break

    assert env.position == 0, "Short position should be closed by TP"
    assert env.realized_pnl > 0, "TP should result in profit"

def test_high_volatility_wipeout():
    """Verify that if price hits both SL and TP in same bar, SL triggers first."""
    # Start at 100. For SHORT: SL at 105, TP at 95.
    # Bar: Open=100, High=110, Low=90, Close=90.
    data = np.array([
        [100, 100, 100, 100, 1000], # Padding
        [100, 110, 90, 90, 1000],   # The Wipeout bar
        [90, 90, 90, 90, 1000],     # More padding
    ], dtype=np.float32)

    sequences = [data]
    keys = ["BTCUSDT_wipeout"]
    stats = {"BTCUSDT": {"mean": [0.0]*5, "std": [1.0]*5}}

    env = TradingEnvironment(
        sequences=sequences, stats=stats, keys=keys, render_mode=None,
        full_seq_len=3, num_features=5, num_actions=4, flat_state_size=0,
        initial_balance=1000.0, pre_signal_len=1, datachannels=['open', 'high', 'low', 'close', 'volume'],
        slippage=0.0, transaction_fee=0.0, agent_session_len=2, agent_history_len=1,
        input_history_len=1, pricechannels=['open', 'high', 'low', 'close'],
        volumechannels=['volume'], otherchannels=[], action_history_len=0,
        inaction_penalty_ratio=0.0, use_risk_management=True,
        stop_loss=0.05, take_profit=0.05, filter_direction="SHORT",
        invert_data=False # Stay on normal chart
    )

    env.reset()
    # Step 0: Open Short at price 100
    env.step(2)
    assert env.position == -1, f"Expected position -1, got {env.position}"
    assert env.real_entry_price == 100

    # Step 1: The volatile bar
    # High is 110 (+10%), Low is 90 (-10%).
    # Both SL (105) and TP (95) are hit.
    _, reward, done, _, info = env.step(0)

    assert env.position == 0, "Position should be closed"
    # SL triggers first, so we should have a LOSS.
    # Entry 100, SL at 105 -> Loss of 5 USDT per unit.
    assert env.realized_pnl < 0, f"SL should trigger first, but got PnL={env.realized_pnl}"
    assert env.balance < 1000.0

def test_mirrored_short_pnl():
    """Verify PnL when invert_data=True and action is 1 (acting as a Short)."""
    # Original data: price falls from 100 to 90.
    data = np.array([
        [100, 100, 100, 100, 1000],
        [90, 90, 90, 90, 1000],
        [80, 80, 80, 80, 1000],
    ], dtype=np.float32)

    sequences = [data]
    keys = ["BTCUSDT_mirrored"]
    # We must provide inverted stats if we want real_price to be -real_original
    stats = {"BTCUSDT": {"mean": [0.0]*5, "std": [1.0]*5}}

    # If we use TradingEnvironment with filter_direction="SHORT" and invert_data=True,
    # it will invert the sequences internally.
    env = TradingEnvironment(
        sequences=sequences, stats=stats, keys=keys, render_mode=None,
        full_seq_len=3, num_features=5, num_actions=4, flat_state_size=0,
        initial_balance=1000.0, pre_signal_len=1, datachannels=['open', 'high', 'low', 'close', 'volume'],
        slippage=0.0, transaction_fee=0.0, agent_session_len=2, agent_history_len=1,
        input_history_len=1, pricechannels=['open', 'high', 'low', 'close'],
        volumechannels=['volume'], otherchannels=[], action_history_len=0,
        inaction_penalty_ratio=0.0, use_risk_management=False,
        filter_direction="SHORT", invert_data=True
    )

    env.reset()
    # In mirrored world, price started at -100 and went to -90 (it rose!)
    # Agent takes action 1 (LONG)
    env.step(1)
    assert env.position == 1
    assert env.direction == "LONG" # Environment sees it as LONG in its mirrored view

    # Let's check how it calculates PnL.
    # real_entry_price will be -100.
    # Next bar real_price will be -90.
    _, reward, done, _, info = env.step(0)

    # PnL for Long: (exit - entry) * volume
    # (-90 - (-100)) * volume = 10 * volume.
    # Volume = 1000 / |-100| = 10.
    # PnL = 10 * 10 = 100.
    # This matches the profit of a SHORT from 100 to 90.
    assert env.realized_pnl > 0
    assert env.balance > 1000.0

def test_action_masking():
    """Verify that action masking correctly remaps and penalizes invalid actions."""
    data = np.ones((10, 5), dtype=np.float32) * 100
    sequences = [data]
    keys = ["BTCUSDT_masking"]
    stats = {"BTCUSDT": {"mean": [0.0]*5, "std": [1.0]*5}}

    # SHORT_ONLY environment on NORMAL chart
    env = TradingEnvironment(
        sequences=sequences, stats=stats, keys=keys, render_mode=None,
        full_seq_len=10, num_features=5, num_actions=4, flat_state_size=0,
        initial_balance=1000.0, pre_signal_len=1, datachannels=['open', 'high', 'low', 'close', 'volume'],
        slippage=0.0, transaction_fee=0.0, agent_session_len=5, agent_history_len=1,
        input_history_len=1, pricechannels=['open', 'high', 'low', 'close'],
        volumechannels=['volume'], otherchannels=[], action_history_len=0,
        inaction_penalty_ratio=0.0, use_risk_management=False,
        filter_direction="SHORT",
        invert_data=False # Normal chart
    )

    env.reset()
    # Try action 1 (LONG) in SHORT_ONLY mode
    _, reward, _, _, _ = env.step(1)

    assert env.position == 0, "Action 1 should have been masked to HOLD"
    assert reward == -0.01, "Should have received masking penalty"

    # LONG_ONLY environment
    env_long = TradingEnvironment(
        sequences=sequences, stats=stats, keys=keys, render_mode=None,
        full_seq_len=10, num_features=5, num_actions=4, flat_state_size=0,
        initial_balance=1000.0, pre_signal_len=1, datachannels=['open', 'high', 'low', 'close', 'volume'],
        slippage=0.0, transaction_fee=0.0, agent_session_len=5, agent_history_len=1,
        input_history_len=1, pricechannels=['open', 'high', 'low', 'close'],
        volumechannels=['volume'], otherchannels=[], action_history_len=0,
        inaction_penalty_ratio=0.0, use_risk_management=False,
        filter_direction="LONG"
    )
    env_long.reset()
    # Try action 2 (SHORT) in LONG_ONLY mode
    _, reward, _, _, _ = env_long.step(2)
    assert env_long.position == 0, "Action 2 should have been masked to HOLD"
    assert reward == -0.01, "Should have received masking penalty"
