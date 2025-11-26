import os
import json
import datetime
import logging
import glob
import numpy as np
import torch
import torch.nn as nn 
from tqdm import tqdm

try:
    from trading_environment import TradingEnvironment
    from agent import D3QN_PER_Agent
    from model import DuelingQNetwork 
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# --- MONKEY PATCH ---
def patched_forward(self, state):
    if state.dim() > 2: state = state.view(state.size(0), -1)
    batch = state.size(0)
    C, L = 10, 90
    history_flat_size = 900
    history_part = state[:, :history_flat_size].contiguous()
    extra_part = state[:, history_flat_size:].contiguous()
    if history_part.numel() != batch * 900:
        raise RuntimeError(f"Slice failed.")
    history_tensor = history_part.view(batch, C, L)
    features = self.feature_extractor(history_tensor)
    features_flat = features.view(batch, -1)
    combined = torch.cat([features_flat, extra_part], dim=1)
    value = self.value_stream(combined)
    advantage = self.advantage_stream(combined)
    q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
    return q_value

DuelingQNetwork.forward = patched_forward

class PerformanceConfig:
    def __init__(self):
        self.use_amp = True; self.amp_dtype = "float16"; self.compile_mode = False; self.compile_dynamic = False

def find_model_checkpoint():
    specific_path = r"output\alpha_seed_404\saved_models\rl_binance_futures_trading_date_20251120_time_015257\best.pth"
    if os.path.exists(specific_path): return specific_path
    default_path = "saved_models/best.pth"
    if os.path.exists(default_path): return default_path
    files = glob.glob("output/**/best.pth", recursive=True)
    if files: return max(files, key=os.path.getmtime)
    return None

MODEL_PATH = find_model_checkpoint()

CONF = {
    "npz_path": "data/val_data_fair_2m.npz", "model_path": MODEL_PATH, "norm_stats_path": "norm_stats.json",
    "num_val_ep": 750,
    "env_params": {
        "full_seq_len": 150, "pre_signal_len": 90, "agent_history_len": 90, "agent_session_len": 60,
        "initial_balance": 10000.0, "transaction_fee": 0.0004, "slippage": 0.00025, "num_actions": 4,
        "inaction_penalty_ratio": 0.001, "backtest_mode": True, "use_risk_management": True,
        "cnn_format": True, "exec_delay_bars": 0, "num_features": 10, "flat_state_size": 912,
        "input_history_len": 90, "action_history_len": 3,
        "data_channels": ["open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_base", "taker_quote", "vwap"],
        "price_channels": ["open", "high", "low", "close"], "volume_channels": ["volume", "quote_volume"],
        "other_channels": ["vwap", "num_trades", "taker_base", "taker_quote"]
    },
    "backtest_kwargs": {
        "stop_loss": 0.01, "take_profit": 0.02, "trailing_stop": 0.018, "trailing_stop_min": 0.005,
        "fee_buffer_mult": 2.0, "delta_p_hysteresis": 0.0015,
    }
}

def load_data_exactly_like_train(npz_path, norm_stats_path):
    logger.info(f"📂 Loading data from {npz_path}...")
    with open(norm_stats_path, 'r') as f: stats = json.load(f)
    means = np.array(stats['mean'], dtype=np.float32); stds = np.array(stats['std'], dtype=np.float32) + 1e-8
    sequences = []
    with np.load(npz_path, allow_pickle=True) as d:
        keys = [k for k in d.files if not k.startswith('_')]
        try: keys.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else x)
        except: keys.sort()
        for key in tqdm(keys, desc="Processing"):
            raw_seq = d[key].astype(np.float32) 
            norm_seq = (raw_seq - means) / stds
            norm_seq = norm_seq.T 
            norm_seq = np.expand_dims(norm_seq, -1)
            sequences.append(norm_seq)
    return sequences, stats

def run_validation():
    if CONF["model_path"] is None: print("❌ 'best.pth' not found!"); return
    seed = 404; torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

    sequences, stats_dict = load_data_exactly_like_train(CONF["npz_path"], CONF["norm_stats_path"])
    limit = min(len(sequences), CONF["num_val_ep"])
    sequences = sequences[:limit]
    
    env_stats = {"means": {}, "stds": {}}
    for i, ch in enumerate(CONF["env_params"]["data_channels"]):
        env_stats["means"][ch] = stats_dict["mean"][i]
        env_stats["stds"][ch] = stats_dict["std"][i]

    logger.info("🔧 Initializing TradingEnvironment...")
    env = TradingEnvironment(sequences=sequences, stats=env_stats, render_mode=None, **CONF["env_params"])

    logger.info("🤖 Initializing Agent...")
    logging.getLogger().setLevel(logging.WARNING) 
    agent = D3QN_PER_Agent(
        state_shape=(10, 90, 1), action_dim=4,
        cnn_maps=[64, 96, 128, 128, 96], cnn_kernels=[3, 3, 3, 3, 3], cnn_strides=[1, 1, 1, 1, 1], cnn_dilations=[1, 2, 4, 8, 16],
        dense_val=[128, 64, 32], dense_adv=[128, 64, 32], additional_feats=12, dropout_model=0.15,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gamma=0.9995, learning_rate=2e-05, batch_size=32, buffer_size=500000, target_update_freq=5000,
        train_start=15000, max_gradient_norm=3.0, per_alpha=0.7, per_beta_start=0.4, per_beta_frames=30000,
        eps_start=1.0, eps_end=0.05, eps_frames=600000, epsilon=0.0, perf_cfg=PerformanceConfig()
    )
    logging.getLogger().setLevel(logging.INFO)
    agent.load_model(CONF["model_path"], strict=True)

    logger.info("🚀 Starting Backtest Validation...")
    
    results = []
    logging.getLogger().setLevel(logging.ERROR)
    pbar = tqdm(range(len(sequences)), desc="Simulating")
    
    for i in pbar:
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        current_dt = start_dt + datetime.timedelta(hours=i)
        
        while not done:
            if isinstance(obs, np.ndarray):
                 if obs.ndim == 3: obs_2d = obs.squeeze(-1) 
                 else: obs_2d = obs 
                 obs_trimmed = obs_2d[:10, :] 
                 history_flat = obs_trimmed.flatten()
                 state_input_flat = np.zeros(912, dtype=np.float32)
                 state_input_flat[:900] = history_flat
                 state_input = state_input_flat[np.newaxis, ...]
            else:
                 state_input = obs

            action = agent.select_action(state_input, training=False)
            current_dt += datetime.timedelta(minutes=1)
            next_obs, reward, terminated, truncated, info = env.backtest_step(
                action=action, signal_dt=current_dt, ticker="ETHUSDT", **CONF["backtest_kwargs"]
            )
            done = terminated or truncated
            obs = next_obs
        
        # DIRECT ATTRIBUTE ACCESS TO FIX 0 TRADES BUG
        results.append({
            "pnl": env.realized_pnl,
            "trades": env.closed_trades
        })
        pbar.set_postfix({"PnL": f"{sum(r['pnl'] for r in results):,.0f}", "Trds": sum(r['trades'] for r in results)})

    logging.getLogger().setLevel(logging.INFO)
    final_pnl = sum(r['pnl'] for r in results)
    final_trades = sum(r['trades'] for r in results)

    print("\n" + "="*44)
    print("📊 FINAL VALIDATION RESULTS")
    print("="*44)
    print(f"Total PnL:      {final_pnl:,.2f}")
    print(f"Total Trades:   {final_trades}")
    print(f"Episodes:       {len(results)}")
    print("="*44)

if __name__ == "__main__":
    run_validation()
