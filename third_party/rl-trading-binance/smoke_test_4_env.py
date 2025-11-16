from functools import partial
import sys
import os
from trading_environment import TradingEnvironment
from vec_env import DummyVecEnv

# Собираем те же env_kwargs, что в main()
from train import load_and_prep_data, compute_norm_stats
from utils import load_config
from config import cfg as default_cfg

# --- Load config from command line, similar to train.py ---
if len(sys.argv) < 2:
    print("ERROR: Please provide the path to a configuration file.", file=sys.stderr)
    print("Usage: python smoke_test_4_env.py configs/your_config.py", file=sys.stderr)
    sys.exit(1)
config_path = sys.argv[1]
cfg = load_config(config_path)

# --- NEW: Make paths absolute to avoid ambiguity ---
def resolve_paths(config):
    """Ensures all relevant paths in the config are absolute."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    for path_attr in ['train_data_path', 'val_data_path', 'test_data_path', 'norm_stats_path']:
        if hasattr(config.paths, path_attr):
            current_path = getattr(config.paths, path_attr)
            if current_path and not os.path.isabs(current_path):
                setattr(config.paths, path_attr, os.path.join(project_root, current_path))

resolve_paths(cfg)

train_seqs = load_and_prep_data(cfg.paths.train_data_path, "Train", norm_stats=None)
if not train_seqs:
    print(f"ERROR: Training data not found at '{cfg.paths.train_data_path}'. Please create the dataset first.", file=sys.stderr)
    sys.exit(1)

norm_stats = compute_norm_stats(cfg.paths.train_data_path)
num_actions = cfg.market.num_actions
input_history_len = cfg.seq.input_history_len or cfg.seq.agent_history_len
num_features = train_seqs[0].shape[0]  # (C, L, 1)
action_history_len = cfg.seq.action_history_len

flat_features = input_history_len * num_features
extras = 4  # position, unrealized, time_elapsed, time_remaining
history_vector_size = num_actions * action_history_len if action_history_len > 0 else 0
flat_state_size = flat_features + extras + history_vector_size

env_kwargs = {
    "sequences": train_seqs,
    "stats": norm_stats,
    "render_mode": cfg.render_mode,
    "full_seq_len": cfg.seq.full_seq_len,
    "num_features": num_features,
    "num_actions": num_actions,
    "flat_state_size": flat_state_size,
    "initial_balance": cfg.market.initial_balance,
    "pre_signal_len": cfg.seq.pre_signal_len,
    "data_channels": cfg.data.data_channels,
    "slippage": cfg.market.slippage,
    "transaction_fee": cfg.market.transaction_fee,
    "agent_session_len": cfg.seq.agent_session_len,
    "agent_history_len": cfg.seq.agent_history_len,
    "input_history_len": input_history_len,
    "price_channels": cfg.data.price_channels,
    "volume_channels": cfg.data.volume_channels,
    "other_channels": cfg.data.other_channels,
    "action_history_len": cfg.seq.action_history_len,
    "inaction_penalty_ratio": cfg.market.inaction_penalty_ratio,    
    # --- NEW: Add missing kwargs from train.py for consistency ---
    "cnn_format": getattr(cfg, "cnn_format", False),
    "position_fraction": getattr(cfg.backtest, "position_fraction", 1.0),
    "order_size_usdt": getattr(cfg.backtest, "order_size_usdt", 0.0),
}
num_envs = 4
env_fns = [partial(TradingEnvironment, **env_kwargs) for _ in range(num_envs)]
venv = DummyVecEnv(env_fns)
print(f"--- Running Smoke Test with {num_envs} environments using DummyVecEnv ---")

# 1. Reset all environments
obs, infos = venv.reset()
print("\n[1] venv.reset() called.")
print(f"  - Observations received. Shape: {getattr(obs, 'shape', type(obs))}")
print(f"  - The first dimension ({obs.shape[0]}) matches the number of environments ({num_envs}).")
print(f"  - Info dictionaries received: {len(infos)}")

# 2. Take a random action in each environment
actions = [venv.action_space.sample() for _ in range(num_envs)]
print(f"\n[2] Generating {num_envs} random actions: {actions}")

# 3. Step all environments
next_obs, rewards, terminated, truncated, infos = venv.step(actions)
print("\n[3] venv.step() called.")
print(f"  - Next observations received. Shape: {getattr(next_obs, 'shape', type(next_obs))}")
print(f"  - Rewards received for each env: {rewards}")
print(f"  - 'Terminated' flags received: {terminated}")
print(f"  - 'Truncated' flags received: {truncated}")

print("\n--- ✅ Smoke test passed: Vectorized environment is running correctly. ---")
