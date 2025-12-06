import os
import sys
import json
import datetime
import logging
import glob
import numpy as np
import torch
import torch.nn as nn 
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

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

class PerformanceConfig:
    def __init__(self):
        self.use_amp = True; self.amp_dtype = "float16"; self.compile_mode = False; self.compile_dynamic = False

def load_config_from_path(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    loader = SourceFileLoader("config_module", config_path)
    config_module = loader.load_module()
    return config_module.cfg

def find_model_checkpoint(cfg=None):
    if cfg and hasattr(cfg, 'paths') and hasattr(cfg.paths, 'model_path') and os.path.exists(cfg.paths.model_path):
        logger.info(f"ℹ️ Using model path from config: {cfg.paths.model_path}")
        return cfg.paths.model_path
    
    model_dir_from_cfg = "."
    if cfg and hasattr(cfg, 'paths') and cfg.paths.model_path:
        model_dir_from_cfg = os.path.dirname(cfg.paths.model_path)

    search_path = os.path.join(model_dir_from_cfg, "best.pth")
    if os.path.exists(search_path):
        return search_path
    
    files = glob.glob(os.path.join(model_dir_from_cfg, "**", "best.pth"), recursive=True)
    if files:
        latest_file = max(files, key=os.path.getmtime)
        logger.info(f"ℹ️ Found latest model checkpoint: {latest_file}")
        return latest_file
    return None

def load_true_config(model_path):
    """Loads the config_train.json from the model's directory."""
    if model_path is None: return None
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config_train.json")
    if os.path.exists(config_path):
        logger.info(f"ℹ️ Loading ground truth config from: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    logger.warning(f"⚠️ config_train.json not found in model directory.")
    return None

def load_and_normalize_data(npz_path, norm_stats_path, paper_symbols_cfg):
    logger.info(f"📂 Loading data from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Data file not found: {npz_path}")
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")

    with open(norm_stats_path, 'r') as f:
        all_stats = json.load(f)

    allowed_assets = paper_symbols_cfg
    if allowed_assets == "ALL":
        allowed_assets = None

    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    valid_keys = []
    
    logger.info(f"Normalizing data for specified symbols: {allowed_assets or 'ALL'}")
    for key in tqdm(data_keys, desc="Normalizing validation data"):
        try:
            asset_name = key.split('_')[0]
        except IndexError:
            continue
        
        if allowed_assets and asset_name not in allowed_assets:
            continue

        asset_specific_stats = all_stats.get(asset_name)
        if asset_specific_stats is None:
            continue

        means = np.array(asset_specific_stats['mean'])
        stds = np.array(asset_specific_stats['std'])
        
        seq = d[key].astype(np.float32)
        if seq.shape[1] != len(means):
            continue
            
        seq = (seq - means) / (stds + 1e-8)
        sequences.append(seq)
        valid_keys.append(key)
    
    d.close()
    if not sequences:
        raise ValueError("No validation sequences were loaded. Check data path and symbol configuration.")
    logger.info(f"Prepared {len(sequences)} validation sequences.")
    
    return sequences, all_stats, valid_keys

def run_validation():
    cli_config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not cli_config_path:
        print("❌ Please provide the path to a config file.")
        return

    user_cfg_module = load_config_from_path(cli_config_path)
    
    model_path = find_model_checkpoint(user_cfg_module)
    if not model_path:
        print("❌ 'best.pth' model file not found.")
        return

    train_cfg_dict = load_true_config(model_path)
    if not train_cfg_dict:
        print("❌ Could not load the ground truth config_train.json from the model's directory.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    norm_stats_path = os.path.join(os.path.dirname(model_path), "norm_stats.json")
    
    val_data_path = train_cfg_dict.get("paths", {}).get("val_data_path", "data/val_data_fair_2m.npz")
    if not os.path.isabs(val_data_path):
        val_data_path = os.path.join(script_dir, val_data_path)

    logger.info(f"ℹ️ Using Validation Data: {val_data_path}")
    logger.info(f"ℹ️ Using Normalization Stats: {norm_stats_path}")

    paper_symbols = train_cfg_dict.get("paper", {}).get("symbols", "ALL")
    sequences, all_stats, keys = load_and_normalize_data(val_data_path, norm_stats_path, paper_symbols)
    
    num_val_ep = train_cfg_dict.get("trainlog", {}).get("num_val_ep", 750)
    limit = min(len(sequences), num_val_ep)
    sequences = sequences[:limit]
    keys = keys[:limit]

    seq_cfg = train_cfg_dict.get("seq", {})
    data_cfg = train_cfg_dict.get("data", {})
    market_cfg = train_cfg_dict.get("market", {})
    backtest_cfg = train_cfg_dict.get("backtest", {})
    model_cfg = train_cfg_dict.get("model", {})
    rl_cfg = train_cfg_dict.get("rl", {})
    per_cfg = train_cfg_dict.get("per", {})
    eps_cfg = train_cfg_dict.get("eps", {})

    input_history_len = seq_cfg.get("input_history_len") or seq_cfg.get("agent_history_len", 90)

    env_params = {
        "full_seq_len": seq_cfg.get("full_seq_len", 150),
        "pre_signal_len": seq_cfg.get("pre_signal_len", 90),
        "agent_history_len": seq_cfg.get("agent_history_len", 90),
        "agent_session_len": seq_cfg.get("agent_session_len", 60),
        "input_history_len": input_history_len,
        "initial_balance": market_cfg.get("initial_balance", 10000.0),
        "transaction_fee": market_cfg.get("transaction_fee", 0.0004),
        "slippage": market_cfg.get("slippage", 0.0002),
        "num_actions": market_cfg.get("num_actions", 4),
        "inaction_penalty_ratio": market_cfg.get("inaction_penalty_ratio", 0.0),
        "backtest_mode": True,
        "use_risk_management": backtest_cfg.get("use_risk_management", False),
        "cnn_format": False,
        "num_features": data_cfg.get("numchannels", 10),
        "action_history_len": seq_cfg.get("action_history_len", 2),
        "datachannels": data_cfg.get("datachannels", []),
        "pricechannels": data_cfg.get("pricechannels", []),
        "volumechannels": data_cfg.get("volumechannels", []),
        "otherchannels": data_cfg.get("otherchannels", []),
    }
    
    flat_state_size = (env_params["agent_history_len"] * env_params["num_features"]) + 4 + (env_params["num_actions"] * env_params["action_history_len"])
    env_params["flat_state_size"] = flat_state_size

    backtest_kwargs = {
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": backtest_cfg.get("trailing_stop"),
        "trailing_stop_min": backtest_cfg.get("trailing_stop_min"),
        "fee_buffer_mult": backtest_cfg.get("fee_buffer_mult"),
        "delta_p_hysteresis": backtest_cfg.get("delta_p_hysteresis"),
    }

    logger.info("🔧 Initializing TradingEnvironment...")
    env = TradingEnvironment(sequences=sequences, stats=all_stats, keys=keys, render_mode=None, **env_params)

    agent = D3QN_PER_Agent(
        state_shape=train_cfg_dict.get("state_shape", [10, 90, 1]),
        action_dim=env_params["num_actions"],
        cnn_maps=model_cfg.get("cnn_maps", []),
        cnn_kernels=model_cfg.get("cnn_kernels", []),
        cnn_strides=model_cfg.get("cnn_strides", []),
        cnn_dilations=model_cfg.get("cnn_dilations", []),
        dense_val=model_cfg.get("dense_val", []),
        dense_adv=model_cfg.get("dense_adv", []),
        additional_feats=model_cfg.get("additional_feats", 12),
        dropout_model=model_cfg.get("dropout_p", 0.0),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gamma=rl_cfg.get("gamma", 0.99),
        learning_rate=rl_cfg.get("lr", 1e-4),
        batch_size=rl_cfg.get("batch_size", 32),
        buffer_size=per_cfg.get("buffer_size", 100000),
        target_update_freq=rl_cfg.get("target_update_freq", 1000),
        train_start=rl_cfg.get("train_start", 1000),
        max_gradient_norm=rl_cfg.get("max_gradient_norm", 1.0),
        per_alpha=per_cfg.get("per_alpha", 0.6),
        per_beta_start=per_cfg.get("per_beta_start", 0.4),
        per_beta_frames=per_cfg.get("per_beta_frames", 100000),
        eps_start=eps_cfg.get("eps_start", 1.0),
        eps_end=eps_cfg.get("eps_end", 0.05),
        eps_frames=eps_cfg.get("eps_decay_frames", 100000),
        epsilon=0.0,
        perf_cfg=PerformanceConfig()
    )
    
    logger.info("🤖 Initializing Agent...")
    agent.load_model(model_path, strict=True)

    logger.info("🚀 Starting Backtest Validation...")
    
    total_pnl = 0
    total_trades = 0
    
    logging.getLogger().setLevel(logging.ERROR)
    pbar = tqdm(range(len(sequences)), desc="Simulating")
    
    for i in pbar:
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        
        ep_trades = 0
        ep_pnl = 0

        signal_dt_for_step = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        ticker_name = "UNKNOWN"
        if keys and i < len(keys):
            try:
                key_parts = keys[i].split('_')
                ticker_name = key_parts[0]
                if len(key_parts) > 1:
                    start_dt_str = key_parts[1]
                    signal_dt_for_step = datetime.datetime.fromisoformat(start_dt_str.replace("Z", "+00:00"))
            except (IndexError, AttributeError, ValueError) as e:
                logger.warning(f"Could not parse ticker/date from key: {keys[i]} due to {e}")

        while not done:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.backtest_step(
                action=action, 
                signal_dt=signal_dt_for_step,
                ticker=ticker_name, 
                **backtest_kwargs
            )
            done = terminated or truncated
            obs = next_obs
            
            if info.get("position_closed", False):
                ep_trades += 1
                ep_pnl += info.get("trade_realized_pnl", 0.0)

        total_trades += ep_trades
        total_pnl += ep_pnl
        
        pbar.set_postfix({"PnL": f"{total_pnl:,.0f}", "Trds": total_trades})

    logging.getLogger().setLevel(logging.INFO)

    print("\n" + "="*44)
    print("📊 FINAL VALIDATION RESULTS")
    print("="*44)
    print(f"Total PnL:      {total_pnl:,.2f}")
    print(f"Total Trades:   {total_trades}")
    print(f"Episodes:       {len(sequences)}")
    print("="*44)

if __name__ == "__main__":
    run_validation()