"""
Cкрипт для валидации производительности ансамбля и одиночных агентов на отложенных данных.
"""
import argparse
import json
import logging
import time
import datetime
import sys
import importlib.util
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
import glob
import random
from collections import defaultdict
from importlib.machinery import SourceFileLoader

# Assuming these are the correct paths from the project structure
from trading_environment import TradingEnvironment
from agent import D3QN_PER_Agent as D3QNPERAgent
from utils import setup_logging

# --- Setup Logging ---
log_file_path = 'output/validation.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnsembleAgent:
    def __init__(self, long_agent_path, short_agent_path, device, agent_creator, use_confidence=False, threshold=0.01, enable_long=True, enable_short=True, verbose=False):
        self.device = device
        self.use_confidence = use_confidence
        self.threshold = threshold
        self.enable_long = enable_long
        self.enable_short = enable_short
        self.verbose = verbose
        self.agent_creator = agent_creator
        
        logger.info(f"🎭 Initializing Ensemble Agent...")
        logger.info(f"  Loading LONG specialist from {long_agent_path}...")
        self.agent_long = self._load_agent(long_agent_path, "LONG")
        
        logger.info(f"  Loading SHORT specialist from {short_agent_path}...")
        self.agent_short = self._load_agent(short_agent_path, "SHORT")
        
        logger.info("✅ Ensemble Agent ready.")

    def _load_agent(self, path, name):
        # Helper to load agent. Assumes D3QN_PER_Agent class structure.
        # We need to reconstruct the agent with the same config as training.
        # Ideally, we load the config from the checkpoint folder.
        
        # For simplicity, we assume the current global cfg matches the agent structure
        # OR we rely on the agent to load its own weights into the architecture created by 'create_agent'
        
        # Load checkpoint to check args if needed (optional)
        # checkpoint = torch.load(path, map_location=self.device)
        
        agent = self.agent_creator(action_dim=3) # Specialists have 3 actions
        agent.load_model(path)
        agent.policy_net.eval()
        return agent

    def _prepare_state(self, state, direction):
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        SEQ_LEN = 90
        NUM_CHANNELS_DATA = 10
        DATA_SIZE = NUM_CHANNELS_DATA * SEQ_LEN

        # Default to returning original state if no feature engineering is needed
        state_prepared = state.copy()

        # This logic is specific to when the environment state has extra features
        # that need to be masked for the specialist agents.
        if state.ndim == 1 and state.shape[0] > DATA_SIZE:
            data_part = state[:DATA_SIZE]
            features_part = state[DATA_SIZE:]

            # Env Features (12): [Pos, Entry, PnL, R_Inv] (4) + [H, L, S, C] * 2 steps (8)
            # Long Agent wants: [Base] + [H, L, C] (Skips S) -> Total 10 features
            # Short Agent wants: [Base] + [H, S, C] (Skips L) -> Total 10 features
            
            if len(features_part) >= 12:
                if direction == "LONG":
                    # Mask to keep: 0-3 (base), 4,5,7 (step1), 8,9,11 (step2)
                    mask = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]
                    feats_prepared = features_part[mask]
                elif direction == "SHORT":
                    # Mask to keep: 0-3 (base), 4,6,7 (step1), 8,10,11 (step2)
                    mask = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
                    feats_prepared = features_part[mask]
                else:
                    # Fallback for safety, though should not be hit
                    feats_prepared = features_part[:10]
                
                state_prepared = np.concatenate([data_part, feats_prepared])
            else:
                # Fallback if feature part is smaller than expected
                state_prepared = np.concatenate([data_part, features_part[:10]])

        return state_prepared

    def get_long_vote(self, state):
        if not self.enable_long: return 0
        
        state_mapped = self._prepare_state(state, "LONG")
        with torch.no_grad():
            t_state = torch.from_numpy(state_mapped).float().unsqueeze(0).to(self.agent_long.device)
            q_values = self.agent_long.policy_net(t_state).squeeze(0)
            
        # Logic: 0=Hold, 1=Open. We IGNORE Close (Index 2).
        if self.use_confidence:
            probs = torch.softmax(q_values, dim=0)
            conf_open = (probs[1] - 0.33).item()
            if conf_open > self.threshold and q_values[1] > q_values[0]:
                return 1
        else:
            # Simple Voting: Open > Hold
            if q_values[1] > q_values[0]:
                return 1
        return 0

    def get_short_vote(self, state):
        if not self.enable_short: return 0

        state_mapped = self._prepare_state(state, "SHORT")

        with torch.no_grad():
            t_state = torch.from_numpy(state_mapped).float().unsqueeze(0).to(self.agent_short.device)
            q_values = self.agent_short.policy_net(t_state).squeeze(0)

        # Logic for 3-action specialist: 0=Hold, 1=Buy, 2=Sell(Short)

        if self.use_confidence:
            probs = torch.softmax(q_values, dim=0)
            conf_open = (probs[2] - 0.33).item()

            if conf_open > self.threshold and q_values[2] > q_values[0]:
                return 2

        else:
            # Simple Voting: Open > Hold
            if q_values[2] > q_values[0]:
                return 2

        return 0

class PerformanceConfig:
    def __init__(self):
        self.use_amp = True; self.amp_dtype = "float16"; self.compile_mode = False; self.compile_dynamic = False

def create_validation_episodes(
    val_sequences, val_keys, num_episodes=750, max_episodes_per_symbol=10, seed=404
):
    if not val_sequences:
        return [], []
    
    episodes_by_symbol = defaultdict(list)
    for i, key in enumerate(val_keys):
        symbol = key.split('_')[0]
        episodes_by_symbol[symbol].append(i)
    
    selected_indices = []
    for symbol, indices in episodes_by_symbol.items():
        n_samples = min(len(indices), max_episodes_per_symbol)
        random.seed(seed)
        selected_indices.extend(random.sample(indices, n_samples))
    
    if len(selected_indices) > num_episodes:
        random.seed(seed)
        final_indices = random.sample(selected_indices, num_episodes)
    else:
        final_indices = selected_indices
        
    random.seed(seed)
    random.shuffle(final_indices)
    
    final_sequences = [val_sequences[i] for i in final_indices]
    final_keys = [val_keys[i] for i in final_indices]
    final_symbols = {val_keys[i].split('_')[0] for i in final_indices}
    
    logging.info(f"Stratified sampling complete. Sampled episodes: {len(final_sequences)}, Symbol coverage: {len(final_symbols)}/{len(episodes_by_symbol)}")
    return final_sequences, final_keys

def load_config_from_path(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    loader = SourceFileLoader("config_module", config_path)
    config_module = loader.load_module()
    return config_module.cfg

def find_model_checkpoint(model_path_arg, cfg=None):
    if model_path_arg and os.path.exists(model_path_arg):
        logger.info(f"ℹ️ Using model path from command line: {model_path_arg}")
        return model_path_arg
    
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
    
    logger.info(f"Applying pre-computed normalization for symbols: {allowed_assets or 'ALL'}")
    
    for key in tqdm(data_keys, desc="Applying normalization"):
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
    parser = argparse.ArgumentParser(description="Validate/test RL agent")
    parser.add_argument("config", type=str, help="Path to config file (e.g. configs/alpha_seed_404_v11.py)")
    parser.add_argument("--model", type=str, help="Path to model checkpoint (for single agent)")
    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help="Validation or test mode")
    
    # Ensemble mode arguments
    parser.add_argument("--ensemble", action='store_true', help="Use ensemble of LONG and SHORT specialists")
    parser.add_argument("--long_model", type=str, help="Path to LONG specialist checkpoint")
    parser.add_argument("--short_model", type=str, help="Path to SHORT specialist checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Ensemble confidence threshold (overrides config)")
    parser.add_argument("--ensemble_verbose", action='store_true', help="Print Q-values during ensemble inference")
    
    args = parser.parse_args()
    
    # --- Load User Config (for paths) ---
    user_cfg_module = load_config_from_path(args.config)
    
    # Auto-fill arguments from config if not provided
    if args.ensemble:
        ensemble_cfg_obj = getattr(user_cfg_module, 'ensemble', None)
        if ensemble_cfg_obj:
            if not args.long_model and hasattr(ensemble_cfg_obj, 'long_model_path'):
                args.long_model = ensemble_cfg_obj.long_model_path
            if not args.short_model and hasattr(ensemble_cfg_obj, 'short_model_path'):
                args.short_model = ensemble_cfg_obj.short_model_path
            
    # Validate arguments
    if args.ensemble:
        if not args.long_model or not args.short_model:
            parser.error("--ensemble requires --long_model and --short_model (via CLI or config)")
    
    if args.model:
        print("⚠️  Warning: --model ignored in ensemble mode")
    elif not args.model and not args.ensemble:
        # In single-agent mode, we can try to find the model automatically
        if hasattr(user_cfg_module.paths, 'model_path'):
             args.model = user_cfg_module.paths.model_path
    
    # Determine the primary model path for loading configs etc.
    # In ensemble mode, we can use the long model as the reference.
    primary_model_path_arg = args.long_model if args.ensemble else args.model
    model_path = find_model_checkpoint(primary_model_path_arg, user_cfg_module)
    
    if not args.ensemble and not model_path:
        print("❌ 'best.pth' model file not found for single agent mode.")
        return
        
    train_cfg_dict = load_true_config(model_path or args.long_model)
    if not train_cfg_dict:
        print("❌ Could not load the ground truth config_train.json from the model's directory.")
        return

    # --- CRITICAL: Use the ground truth config for all parameters ---
    cfg = train_cfg_dict
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For ensemble, norm_stats could be different. Assume they are the same and load from long_model path or config
    ensemble_cfg_obj = getattr(user_cfg_module, 'ensemble', None)
    if ensemble_cfg_obj and hasattr(ensemble_cfg_obj, 'norm_stats_path') and os.path.exists(ensemble_cfg_obj.norm_stats_path):
        norm_stats_path = ensemble_cfg_obj.norm_stats_path
    elif hasattr(user_cfg_module.paths, 'norm_stats_path') and os.path.exists(user_cfg_module.paths.norm_stats_path):
        norm_stats_path = user_cfg_module.paths.norm_stats_path
    else:
        norm_stats_path = os.path.join(os.path.dirname(model_path or args.long_model), "norm_stats.json")
    
    if hasattr(user_cfg_module.paths, 'val_data_path'):
        val_data_path = user_cfg_module.paths.val_data_path
    else:
        val_data_path = cfg.get("paths", {}).get("val_data_path", "data/val_data_fair_2m.npz")
    
    if not os.path.isabs(val_data_path):
        val_data_path = os.path.join(script_dir, val_data_path)
        
    logger.info(f"ℹ️ Using Validation Data: {val_data_path}")
    logger.info(f"ℹ️ Using Normalization Stats: {norm_stats_path}")
    
    paper_symbols = cfg.get("paper", {}).get("symbols", "ALL")
    sequences, all_stats, keys = load_and_normalize_data(val_data_path, norm_stats_path, paper_symbols)
    
    trainlog_cfg = cfg.get("trainlog", {})
    
    sequences, keys = create_validation_episodes(
        val_sequences=sequences,
        val_keys=keys,
        num_episodes=trainlog_cfg.get("num_val_ep", 750), # Use num_val_ep from the training config
        seed=cfg.get("random_seed", 404)
    )
    
    seq_cfg = cfg.get("seq", {})
    data_cfg = cfg.get("data", {})
    market_cfg = cfg.get("market", {})
    backtest_cfg = cfg.get("backtest", {})
    model_cfg = cfg.get("model", {})
    rl_cfg = cfg.get("rl", {})
    per_cfg = cfg.get("per", {})
    eps_cfg = cfg.get("eps", {})
    
    # --- Get Ensemble Params ---
    ensemble_cfg = getattr(user_cfg_module, 'ensemble', None)
    threshold_val = 0.02 # Default
    disable_cross_close = False # Default
    conflict_cooldown_bars = 0 # Default
    if args.threshold is not None:
        threshold_val = args.threshold
    elif ensemble_cfg and hasattr(ensemble_cfg, 'threshold'):
        threshold_val = ensemble_cfg.threshold
        
    if ensemble_cfg and hasattr(ensemble_cfg, 'disable_cross_close'):
        disable_cross_close = ensemble_cfg.disable_cross_close
        if disable_cross_close:
            logger.info("ℹ️ Cross-closing logic is DISABLED by config.")

    if ensemble_cfg and hasattr(ensemble_cfg, 'conflict_cooldown_bars'):
        conflict_cooldown_bars = ensemble_cfg.conflict_cooldown_bars
        if conflict_cooldown_bars > 0:
            logger.info(f"ℹ️ Conflict cooldown is ENABLED: {conflict_cooldown_bars} bars.")

    # --- Env Params ---
    # Retrieve base parameters from config sections
    num_channels = cfg.get("num_channels", 10)
    
    # FIX: Generate list of channel names required by TradingEnvironment
    # It needs to find "close" in this list.
    default_datachannels = ['open', 'high', 'low', 'close', 'volume']
    if num_channels > 5:
        default_datachannels += [f"feat_{i}" for i in range(5, num_channels)]

    env_num_actions = market_cfg.get("num_actions", 3)
    env_params = {
        "sequences": sequences,
        "stats": all_stats,
        "keys": keys,
        "render_mode": None,
        
        # --- Missing Required Arguments for TradingEnvironment ---
        "full_seq_len": seq_cfg.get("full_seq_len", 150),
        "num_features": num_channels,
        "flat_state_size": 0, 
        "initial_balance": market_cfg.get("initial_balance", 10000.0),
        "pre_signal_len": seq_cfg.get("pre_signal_len", 90),
        
        # FIX: datachannels must be a LIST of strings, containing "close"
        "datachannels": data_cfg.get("datachannels", default_datachannels),
        
        "agent_session_len": seq_cfg.get("agent_session_len", 60),
        "agent_history_len": seq_cfg.get("agent_history_len", 90),
        "input_history_len": seq_cfg.get("input_history_len", 90),
        
        # Channel definitions (Indices)
        "pricechannels": [0, 1, 2, 3], 
        "volumechannels": [4],
        "otherchannels": list(range(5, num_channels)),
        
        "action_history_len": seq_cfg.get("action_history_len", 2),
        "inaction_penalty_ratio": market_cfg.get("inaction_penalty_ratio", 0.0),
        "backtest_mode": True,

        # --- Standard Params ---
        # Must match training action space (alpha_seed_404_v11.py sets 3 actions).
        "num_actions": env_num_actions,
        "allowed_directions": market_cfg.get("allowed_directions", ['LONG', 'SHORT']),
        "filter_direction": None, # CRITICAL: Do not filter here for ensemble
        "transaction_fee": market_cfg.get("transaction_fee", 0.0004),
        "slippage": market_cfg.get("slippage", 0.0002),
        "position_fraction": market_cfg.get("position_fraction", 0.1)
    }

    env_long, env_short, env = None, None, None
    logger.info("🌍 Initializing TradingEnvironment(s)...")

    if args.ensemble:
        # Create separate envs for LONG and SHORT
        # IMPORTANT: Do not use filter_direction. Use allowed_directions to constrain agent.
        # This ensures that the number of sequences remains the same for both environments,
        # preventing the IndexError when using forced_index.
        
        env_params_long = env_params.copy()
        env_params_long["allowed_directions"] = ['LONG']
        env_params_long["num_actions"] = 4 # 4 actions needed for cross-closing logic (action 3)
        env_long = TradingEnvironment(**env_params_long)
        
        env_params_short = env_params.copy()
        env_params_short["allowed_directions"] = ['SHORT']
        env_params_short["num_actions"] = 4 # 4 actions needed for cross-closing logic (action 3)
        env_short = TradingEnvironment(**env_params_short)
        
        # Add an assertion to catch data mismatch early
        assert len(env_long.sequences) == len(env_short.sequences), \
            f"Sequence count mismatch: LONG ({len(env_long.sequences)}) vs SHORT ({len(env_short.sequences)})"
            
        logger.info("  -> LONG and SHORT environments created for ensemble.")
    else:
        # Single agent mode
        env = TradingEnvironment(**env_params)
        logger.info("  -> Single environment created for single agent mode.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Using device: {device}")
    
    # --- Helper to create/load agent ---
    def create_agent(action_dim, additional_feats_override=None):
        true_add_feats = additional_feats_override or model_cfg.get("additional_feats", 12)
        
        return D3QNPERAgent(
            state_shape=seq_cfg.get("state_shape", (10, 90, 1)),
            action_dim=action_dim,
            cnn_maps=model_cfg.get("cnn_maps"),
            cnn_kernels=model_cfg.get("cnn_kernels"),
            cnn_strides=model_cfg.get("cnn_strides"),
            cnn_dilations=model_cfg.get("cnn_dilations"),
            dense_val=model_cfg.get("dense_val"),
            dense_adv=model_cfg.get("dense_adv"),
            additional_feats=true_add_feats,
            # dropout_p=model_cfg.get("dropout_p", 0.0), # Removed as requested by error
            device=device,
            gamma=rl_cfg.get("gamma", 0.99),
            learning_rate=rl_cfg.get("lr", 1e-4),
            batch_size=rl_cfg.get("batch_size", 32),
            buffer_size=100, # Small buffer for val
            perf_cfg=PerformanceConfig(),
            
            # --- Missing Args from Error Message ---
            dropout_model=model_cfg.get("dropout_p", 0.0), # Maybe called dropout_model?
            target_update_freq=rl_cfg.get("target_update_freq", 1000),
            train_start=rl_cfg.get("train_start", 1000),
            per_alpha=per_cfg.get("alpha", 0.6),
            per_beta_start=per_cfg.get("beta_start", 0.4),
            per_beta_frames=per_cfg.get("beta_frames", 10000),
            eps_start=eps_cfg.get("eps_start", 1.0),
            eps_end=eps_cfg.get("eps_end", 0.01),
            eps_frames=eps_cfg.get("eps_frames", 10000),
            epsilon=eps_cfg.get("eps_start", 1.0), # Initial epsilon
            max_gradient_norm=rl_cfg.get("max_gradient_norm", 1.0)
        )

    agent = None
    if args.ensemble:
        ensemble_cfg = getattr(user_cfg_module, 'ensemble', None)
        use_conf = getattr(ensemble_cfg, 'use_confidence', False) if ensemble_cfg else False
        enable_long = getattr(ensemble_cfg, 'enable_long', True) if ensemble_cfg else True
        enable_short = getattr(ensemble_cfg, 'enable_short', True) if ensemble_cfg else True

        agent = EnsembleAgent(
            long_agent_path=args.long_model,
            short_agent_path=args.short_model,
            device=device,
            agent_creator=create_agent,
            use_confidence=use_conf,
            threshold=threshold_val,
            enable_long=enable_long,
            enable_short=enable_short,
            verbose=args.ensemble_verbose
        )
        
    else:
        # Single Agent Init
        print(f"\n📦 Loading single agent model from: {model_path}")
        
        def get_specialist_action_dim(model_path):
            # Hack/Heuristic to determine action dim if not in config
            # But usually 3
            return 3

        num_actions_env = 3 # Default for single
        if "SHORT_ONLY" in model_path or "LONG_ONLY" in model_path:
             num_actions_env = 3
        
        action_history_len = seq_cfg.get("action_history_len", 2)
        true_additional_feats = 4 + (num_actions_env * action_history_len)
        
        agent = create_agent(action_dim=num_actions_env, additional_feats_override=true_additional_feats)
        agent.load_model(model_path, strict=True)
        agent.policy_net.eval()
        print("✅ Model loaded successfully")

    logger.info("🚀 Starting Backtest Validation...")
    
    all_trades = []
    total_bars_processed = 0
    start_time = time.time()
    
    logging.getLogger().setLevel(logging.ERROR)
    
    pbar = tqdm(range(len(sequences)), desc="Simulating")
    
    for i in pbar:
        # --- Ticker and Datetime Setup ---
        signal_dt = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        ticker_name = "UNKNOWN"
        try:
            key_parts = keys[i].split('_')
            ticker_name = key_parts[0]
            if len(key_parts) > 1:
                start_dt_str = key_parts[1]
                signal_dt = datetime.datetime.fromisoformat(start_dt_str.replace("Z", "+00:00"))
        except (IndexError, AttributeError, ValueError):
            pass

        if args.ensemble:
            # Add a safeguard for index bounds.
            if i >= len(env_long.sequences) or i >= len(env_short.sequences):
                logger.warning(f"Skipping episode index {i} as it is out of bounds for an environment.")
                continue

            # --- Ensemble Mode Simulation ---
            obs_l, info_l = env_long.reset(options={"forced_index": i})
            obs_s, info_s = env_short.reset(options={"forced_index": i})
            done_l, done_s = False, False
            cooldown_until_step = 0

            while not (done_l and done_s):
                current_step = env_long.step_idx  # Or env_short, they are in sync

                # --- Cooldown Logic ---
                if current_step < cooldown_until_step:
                    # print(f"[{ticker_name}] Step {current_step}: Cooldown active until step {cooldown_until_step}. Forcing HOLD.")
                    vote_l, vote_s = 0, 0
                else:
                    # 1. GET VOTES (0, 1, or 2) - No Close actions
                    vote_l = 0
                    if not done_l:
                        vote_l = agent.get_long_vote(obs_l)
                    
                    vote_s = 0
                    if not done_s:
                        vote_s = agent.get_short_vote(obs_s)
                    
                # 2. CONFLICT RESOLUTION & CROSS-CLOSING
                final_act_l = vote_l
                final_act_s = vote_s

                if not disable_cross_close:
                    long_wants_open = (vote_l == 1)
                    short_wants_open = (vote_s == 2)

                    long_is_active = (env_long.position > 0)
                    short_is_active = (env_short.position < 0)

                    # Сценарий 1: Прямой конфликт (оба хотят войти одновременно)
                    if long_wants_open and short_wants_open:
                        if long_is_active:
                            # Long активен, закрываем его и даем Short открыть позицию.
                            print(f"[{ticker_name}] Event: Conflict vote. Closing active Long and opening Short.")
                            final_act_l = 3
                            final_act_s = 2 # Разрешаем Short открыть
                        elif short_is_active:
                            # Short активен, закрываем его и даем Long открыть позицию.
                            print(f"[{ticker_name}] Event: Conflict vote. Closing active Short and opening Long.")
                            final_act_s = 3
                            final_act_l = 1 # Разрешаем Long открыть
                        else:
                            # Никто не активен. Ничего не делаем.
                            print(f"[{ticker_name}] Event: Conflict vote. No active positions. Holding.")
                            final_act_l = 0
                            final_act_s = 0
                        
                        # Cooldown применяется в любом случае конфликта.
                        cooldown_until_step = current_step + conflict_cooldown_bars

                    # Сценарий 2: Нет прямого конфликта, проверяем перекрестное закрытие
                    else:
                        # Если Long хочет войти И Short УЖЕ в сделке -> закрываем Short
                        if long_wants_open and short_is_active:
                            print(f"[{ticker_name}] Event: Long vote closes existing Short position.")
                            final_act_s = 3   # Принудительно закрыть Short
                            final_act_l = 0   # Long-агент должен ждать, его сигнал был использован для закрытия Short

                        # Если Short хочет войти И Long УЖЕ в сделке -> закрываем Long
                        elif short_wants_open and long_is_active:
                            print(f"[{ticker_name}] Event: Short vote closes existing Long position.")
                            final_act_l = 3   # Принудительно закрыть Long
                            final_act_s = 0   # Short-агент должен ждать, его сигнал был использован для закрытия Long
                
                # 3. EXECUTION
                # -- Long Env --
                if not done_l:
                    next_obs_l, _, term_l, trunc_l, info_l = env_long.backtest_step(
                        action=final_act_l, signal_dt=signal_dt, ticker=ticker_name
                    )
                    obs_l = next_obs_l
                    done_l = term_l or trunc_l
                    
                    if info_l.get('position_closed'):
                        t_data = info_l.copy()
                        t_data['symbol'] = f"{ticker_name}_L"
                        t_data['pnl'] = t_data.get('trade_realized_pnl', 0)
                        t_data['commission'] = t_data.get('trade_commission', 0)
                        t_data['net_pnl'] = t_data['pnl'] - t_data['commission']
                        t_data['bars'] = t_data.get('holding_duration_bars', 0)
                        all_trades.append(t_data)
                        
                # -- Short Env --
                if not done_s:
                    next_obs_s, _, term_s, trunc_s, info_s = env_short.backtest_step(
                        action=final_act_s, signal_dt=signal_dt, ticker=ticker_name
                    )
                    obs_s = next_obs_s
                    done_s = term_s or trunc_s
                    
                    if info_s.get('position_closed'):
                        t_data = info_s.copy()
                        t_data['symbol'] = f"{ticker_name}_S"
                        t_data['pnl'] = t_data.get('trade_realized_pnl', 0)
                        t_data['commission'] = t_data.get('trade_commission', 0)
                        t_data['net_pnl'] = t_data['pnl'] - t_data['commission']
                        t_data['bars'] = t_data.get('holding_duration_bars', 0)
                        all_trades.append(t_data)

                total_bars_processed += 1
        else:
            # --- Single Agent Mode Simulation ---
            obs, _ = env.reset(options={"forced_index": i})
            done = False
            
            while not done:
                action = agent.select_action(obs, training=False)
                
                next_obs, reward, terminated, truncated, info = env.backtest_step(
                    action=action,
                    signal_dt=signal_dt,
                    ticker=ticker_name
                )
                
                if info.get('position_closed'):
                    pnl = info["trade_realized_pnl"]
                    comm = info.get("trade_commission", 0.0)
                    net_pnl = pnl - comm
                    
                    trade_data = {
                        "symbol": ticker_name,
                        "direction": info.get("direction", "UNKNOWN"),
                        "pnl": pnl,
                        "net_pnl": net_pnl,
                        "commission": comm,
                        "bars": info.get("holding_duration_bars", 0),
                        "tsl_triggered": info.get('tsl_triggered', False)
                    }
                    all_trades.append(trade_data)
                    
                obs = next_obs
                done = terminated or truncated
                total_bars_processed += 1

        
        # --- Progress Bar Update ---
        pbar.set_postfix({
            "PnL": f"{sum(t.get('net_pnl', 0.0) for t in all_trades):,.0f}",
            "Trds": len(all_trades)
        })

    logging.getLogger().setLevel(logging.INFO)

    # --- Metrics Calculation ---
    total_duration = time.time() - start_time
    total_trades = len(all_trades)
    win_count = sum(1 for t in all_trades if t.get('net_pnl', 0.0) > 0)
    loss_count = total_trades - win_count
    wr_ratio = win_count / max(1, total_trades)
    
    gross_pnl = sum(t.get('pnl', 0.0) for t in all_trades)
    net_pnl = sum(t.get('net_pnl', 0.0) for t in all_trades)
    total_commission = sum(t.get('commission', 0.0) for t in all_trades)
    avg_pnl_per_trade = net_pnl / max(1, total_trades)
    
    trade_pnls = [t.get('net_pnl', 0.0) for t in all_trades]
    best_trade = max(trade_pnls) if trade_pnls else 0.0
    worst_trade = min(trade_pnls) if trade_pnls else 0.0
    
    long_trades = sum(1 for t in all_trades if t.get('direction') == 'LONG')
    short_trades = sum(1 for t in all_trades if t.get('direction') == 'SHORT')
    
    holding_times = [t.get('bars', 0) for t in all_trades]
    avg_holding_time = np.mean(holding_times) if holding_times else 0.0
    max_holding_time = max(holding_times) if holding_times else 0.0
    min_holding_time = min(holding_times) if holding_times else 0.0
    
    bars_per_day = 1440
    trading_time_days = total_bars_processed / bars_per_day if bars_per_day > 0 else 0.0
    pnl_per_day = net_pnl / max(1, trading_time_days)
    
    initial_balance = env.initial_balance if hasattr(env, 'initial_balance') else 10000
    roi_percent = (net_pnl / initial_balance) * 100
    roi_annualized = roi_percent * (365.0 / trading_time_days) if trading_time_days > 0 else 0.0
    
    pos_pnls = [p for p in trade_pnls if p > 0]
    neg_pnls = [p for p in trade_pnls if p < 0]
    avg_win_size = np.mean(pos_pnls) if pos_pnls else 0.0
    avg_loss_size = np.mean(neg_pnls) if neg_pnls else 0.0
    win_loss_ratio = abs(avg_win_size / avg_loss_size) if avg_loss_size != 0 else float('inf')
    
    expectancy = (wr_ratio * avg_win_size) + ((1 - wr_ratio) * avg_loss_size)
    profit_factor = sum(pos_pnls) / max(1e-9, abs(sum(neg_pnls)))
    
    equity_curve = np.cumsum([initial_balance] + trade_pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    returns = np.array(trade_pnls) / initial_balance
    if len(returns) > 1:
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        downside_std = np.std(returns[returns < 0], ddof=1) if len(returns[returns < 0]) > 1 else 1e-9
        sharpe = (mean_r / std_r) if std_r > 1e-9 else 0.0
        sortino = (mean_r / downside_std) if downside_std > 1e-9 else 0.0
    else:
        sharpe, sortino = 0.0, 0.0
        
    tsl_hits = sum(1 for t in all_trades if t.get('tsl_triggered', False))

    print("\n" + "="*44)
    print("📊 FINAL VALIDATION RESULTS")
    print("="*44)
    if args.ensemble:
        print(f"Mode: ENSEMBLE (LONG + SHORT specialists) | Thresh: {threshold_val}")
    else:
        print(f"Mode: Single agent")
    print(f"Trades: {total_trades} (Long: {long_trades}, Short: {short_trades}, Win: {win_count}, Loss: {loss_count}) | WinRate: {wr_ratio:.2%} | PF: {profit_factor:.4f}")
    print(f"Gross PnL: {gross_pnl:.2f} | Net PnL: {net_pnl:.2f} | Commission: {total_commission:.2f} | Avg/Trade: {avg_pnl_per_trade:.2f}")
    print(f"Best Trade: {best_trade:+.2f} | Worst Trade: {worst_trade:+.2f} | MaxDD: {abs(max_dd):.2%} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f}")
    print(f"Avg Hold: {avg_holding_time:.2f} bars | Min Hold: {min_holding_time} bars | Max Hold: {max_holding_time} bars")
    print(f"Duration: {total_duration:.2f}s | Bars: {total_bars_processed} | Trading Days: {trading_time_days:.1f}")
    print(f"PnL/Day: {pnl_per_day:.2f} USDT | ROI: {roi_percent:.2f}% | Annualized ROI: {roi_annualized:.1f}%")
    print(f"Commission: {(total_commission / max(1e-9, abs(gross_pnl)))*100:.1f}% of gross | Avg Win: {avg_win_size:.2f} | Avg Loss: {avg_loss_size:.2f} | W/L Ratio: {win_loss_ratio:.2f}")
    print(f"Expectancy/Trade: {expectancy:.2f} USDT")
    print(f"TSL hits: {tsl_hits} ({tsl_hits/max(1, total_trades):.2%})")
    print("="*44)

if __name__ == "__main__":
    run_validation()