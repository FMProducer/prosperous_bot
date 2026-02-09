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
    env.reset(options={"forced_index": 0})

    # Step 0: Open Short
    env.step(2)

    # Step 1: Trigger RM logic once
    env.step(0)
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
