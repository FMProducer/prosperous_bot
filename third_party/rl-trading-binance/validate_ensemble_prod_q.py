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
# Оптимизация для Ryzen: 1 поток на модель в ансамбле 
# предотвращает борьбу за L3 кэш
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
import glob
import random
from collections import defaultdict
from importlib.machinery import SourceFileLoader
import types

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
    def __init__(self, long_agent_path, short_agent_path, device, agent_creator, use_confidence=False, long_threshold=0.01, short_threshold=0.01, enable_long=True, enable_short=True, verbose=False):
        self.device = device
        self.use_confidence = use_confidence
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.enable_long = enable_long
        self.enable_short = enable_short
        self.verbose = verbose
        self.agent_creator = agent_creator
        
        logger.info(f"🎭 Initializing Ensemble Agent...")
        logger.info(f"   Loading LONG specialist from {long_agent_path}...")
        self.agent_long = self._load_agent(long_agent_path, "LONG")
        
        logger.info(f"   Loading SHORT specialist from {short_agent_path}...")
        self.agent_short = self._load_agent(short_agent_path, "SHORT")
        logger.info("✅ Ensemble Agent ready.")

    def _load_agent(self, path, name):
        agent = self.agent_creator(action_dim=3) 
        agent.load_model(path)
        agent.policy_net.eval()
        return agent

    def _prepare_state(self, state, direction):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        SEQ_LEN = 90
        NUM_CHANNELS_DATA = 10
        DATA_SIZE = NUM_CHANNELS_DATA * SEQ_LEN
        
        state_prepared = state.copy()
        
        if state.ndim == 1 and state.shape[0] > DATA_SIZE:
            data_part = state[:DATA_SIZE]
            features_part = state[DATA_SIZE:]
            
            if len(features_part) >= 12:
                if direction == "LONG":
                     mask = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]
                     feats_prepared = features_part[mask]
                elif direction == "SHORT":
                     mask = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
                     feats_prepared = features_part[mask]
                else:
                     feats_prepared = features_part[:10]
                state_prepared = np.concatenate([data_part, feats_prepared])
            else:
                state_prepared = np.concatenate([data_part, features_part[:10]])
                
        return state_prepared

    def get_long_vote(self, state):
        if not self.enable_long:
            return False, 0.0
        state_mapped = self._prepare_state(state, "LONG")
        with torch.no_grad():
            t_state = torch.from_numpy(state_mapped).float().unsqueeze(0).to(self.agent_long.device)
            q_values = self.agent_long.policy_net(t_state).squeeze(0)
            wants_to_open = (q_values[1] > q_values[0]).item()
            if self.use_confidence:
                probs = torch.softmax(q_values, dim=0)
                confidence = (probs[1] - 0.33).item()
            else:
                confidence = 1.0 
            return wants_to_open, confidence

    def get_short_vote(self, state):
        if not self.enable_short:
            return False, 0.0
        state_mapped = self._prepare_state(state, "SHORT")
        with torch.no_grad():
            t_state = torch.from_numpy(state_mapped).float().unsqueeze(0).to(self.agent_short.device)
            q_values = self.agent_short.policy_net(t_state).squeeze(0)
            wants_to_open = (q_values[2] > q_values[0]).item()
            if self.use_confidence:
                probs = torch.softmax(q_values, dim=0)
                confidence = (probs[2] - 0.33).item()
            else:
                confidence = 1.0 
            return wants_to_open, confidence

class PerformanceConfig:
    def __init__(self):
        self.use_amp = True
        self.amp_dtype = "float16"
        self.compile_mode = False
        self.compile_dynamic = False

def create_validation_episodes(val_sequences, val_keys, num_episodes=750, max_episodes_per_symbol=10, seed=404):
    if not val_sequences: return [], []
    
    # Group indices by symbol
    episodes_by_symbol = defaultdict(list)
    for i, key in enumerate(val_keys):
        symbol = key.split('_')[0]
        episodes_by_symbol[symbol].append(i)

    selected_indices = []
    
    # 1. Select episodes respecting the per-symbol limit
    for symbol, indices in episodes_by_symbol.items():
        # If max_episodes_per_symbol is huge (e.g. 5000), this effectively takes ALL episodes for the symbol
        n_samples = min(len(indices), max_episodes_per_symbol)
        
        # We want deterministic sampling if possible, or random if downsampling
        random.seed(seed)
        selected_indices.extend(random.sample(indices, n_samples))

    # 2. Select final list respecting the total global limit
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
    logger.info(f"Limits used: Total Ep={num_episodes}, Max/Sym={max_episodes_per_symbol}")
    
    return final_sequences, final_keys

def load_config_from_path(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config_module = types.ModuleType("user_config_module")
    config_globals = config_module.__dict__
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            code = f.read()
            exec(code, config_globals)
    except Exception as e:
        logger.error(f"Failed to execute config file: {e}")
        raise e
    return config_module

def find_model_checkpoint(model_path_arg, cfg=None):
    if model_path_arg and os.path.exists(model_path_arg):
        logger.info(f"ℹ️ Using model path from command line: {model_path_arg}")
        return model_path_arg
    true_cfg = cfg
    if hasattr(cfg, 'cfg'): 
        true_cfg = cfg.cfg
    if true_cfg and hasattr(true_cfg, 'paths') and hasattr(true_cfg.paths, 'model_path') and os.path.exists(true_cfg.paths.model_path):
        logger.info(f"ℹ️ Using model path from config: {true_cfg.paths.model_path}")
        return true_cfg.paths.model_path
    model_dir_from_cfg = "."
    if true_cfg and hasattr(true_cfg, 'paths') and true_cfg.paths.model_path:
        model_dir_from_cfg = os.path.dirname(true_cfg.paths.model_path)
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
    if allowed_assets == "ALL": allowed_assets = None
    d = np.load(npz_path, allow_pickle=True)
    data_keys = [k for k in d.files if not k.startswith('_')]
    sequences = []
    valid_keys = []
    logger.info(f"Applying pre-computed normalization for symbols: {allowed_assets or 'ALL'}")
    for key in tqdm(data_keys, desc="Applying normalization"):
        try:
            asset_name = key.split('_')[0]
        except IndexError: continue
        if allowed_assets and asset_name not in allowed_assets: continue
        asset_specific_stats = all_stats.get(asset_name)
        if asset_specific_stats is None: continue
        means = np.array(asset_specific_stats['mean'])
        stds = np.array(asset_specific_stats['std'])
        seq = d[key].astype(np.float32)
        if seq.shape[1] != len(means): continue
        seq = (seq - means) / (stds + 1e-8)
        sequences.append(seq)
        valid_keys.append(key)
    d.close()
    if not sequences: raise ValueError("No validation sequences were loaded.")
    logger.info(f"Prepared {len(sequences)} validation sequences.")
    return sequences, all_stats, valid_keys

def run_validation():
    parser = argparse.ArgumentParser(description="Validate/test RL agent")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help="Validation or test mode")
    parser.add_argument("--ensemble", action='store_true', help="Use ensemble")
    parser.add_argument("--long_model", type=str, help="Path to LONG specialist checkpoint")
    parser.add_argument("--short_model", type=str, help="Path to SHORT specialist checkpoint")
    parser.add_argument("--long-threshold", type=float, default=None, help="Ensemble confidence threshold for LONG")
    parser.add_argument("--short-threshold", type=float, default=None, help="Ensemble confidence threshold for SHORT")
    parser.add_argument("--ensemble_verbose", action='store_true', help="Print Q-values")
    
    args = parser.parse_args()
    
    print("!!! Я ТОЧНО ЗАПУСТИЛСЯ: FULL COVERAGE VERSION (FIXED CFG) !!!")
    logger.error("!!! Я ТОЧНО ЗАПУСТИЛСЯ: FULL COVERAGE VERSION (FIXED CFG) !!!")

    user_cfg_module = load_config_from_path(args.config)
    user_cfg_obj = getattr(user_cfg_module, 'cfg', None)
    if user_cfg_obj is None: 
        logger.error("❌ 'cfg' object not found in user config module!")
    
    # --- FIX: Securely extract ensemble config from the user's Python file FIRST ---
    ensemble_settings = None
    if args.ensemble:
        if user_cfg_obj and hasattr(user_cfg_obj, 'ensemble'):
            ensemble_settings = getattr(user_cfg_obj, 'ensemble')
            logger.info("✅ Successfully loaded ENSEMBLE settings from Python config.")
        else:
            logger.warning("⚠️ Could not find 'ensemble' configuration block in the provided Python config file.")

        # Pre-fill model paths from the securely loaded ensemble config if they exist
        if ensemble_settings:
            if not args.long_model and hasattr(ensemble_settings, 'long_model_path'):
                args.long_model = ensemble_settings.long_model_path
            if not args.short_model and hasattr(ensemble_settings, 'short_model_path'):
                args.short_model = ensemble_settings.short_model_path

    if args.ensemble:
        if not args.long_model or not args.short_model:
            parser.error("--ensemble requires --long_model and --short_model paths, either via arguments or in the config file.")
        if args.model: 
            print("⚠️ Warning: --model ignored in ensemble mode")
    elif not args.model and not args.ensemble:
        if user_cfg_obj and hasattr(user_cfg_obj, 'paths') and hasattr(user_cfg_obj.paths, 'model_path'):
            args.model = user_cfg_obj.paths.model_path
    
    primary_model_path_arg = args.long_model if args.ensemble else args.model
    model_path = find_model_checkpoint(primary_model_path_arg, user_cfg_module)
    
    if not args.ensemble and not model_path:
        print("❌ 'best.pth' model file not found for single agent mode.")
        return

    train_cfg_dict = load_true_config(model_path or args.long_model)
    if not train_cfg_dict:
        print("❌ Could not load the ground truth config_train.json.")
        return

    # This is where the original cfg object is overwritten.
    # Our ensemble_settings are now safe.
    cfg = train_cfg_dict
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use the securely loaded ensemble settings
    ensemble_cfg_obj = ensemble_settings

    if ensemble_cfg_obj and hasattr(ensemble_cfg_obj, 'norm_stats_path') and os.path.exists(ensemble_cfg_obj.norm_stats_path):
        norm_stats_path = ensemble_cfg_obj.norm_stats_path
    elif user_cfg_obj and hasattr(user_cfg_obj, 'paths') and hasattr(user_cfg_obj.paths, 'norm_stats_path') and os.path.exists(user_cfg_obj.paths.norm_stats_path):
        norm_stats_path = user_cfg_obj.paths.norm_stats_path
    else:
        norm_stats_path = os.path.join(os.path.dirname(model_path or args.long_model), "norm_stats.json")

    if user_cfg_obj and hasattr(user_cfg_obj, 'paths') and hasattr(user_cfg_obj.paths, 'val_data_path'):
        val_data_path = user_cfg_obj.paths.val_data_path
    else:
        val_data_path = cfg.get("paths", {}).get("val_data_path", "data/val_data_fair_2m.npz")

    if not os.path.isabs(val_data_path): val_data_path = os.path.join(script_dir, val_data_path)

    logger.info(f"ℹ️ Using Validation Data: {val_data_path}")
    logger.info(f"ℹ️ Using Normalization Stats: {norm_stats_path}")

    paper_symbols = cfg.get("paper", {}).get("symbols", "ALL")
    sequences, all_stats, keys = load_and_normalize_data(val_data_path, norm_stats_path, paper_symbols)
    
    trainlog_cfg = cfg.get("trainlog", {})
    
    # === UPDATED SAMPLING LOGIC (FIXED) ===
    # 1. Try to get total episodes from cfg
    total_val_ep = trainlog_cfg.get("num_val_ep", 750) # Default from train config
    
    user_trainlog_cfg = getattr(user_cfg_obj, 'trainlog', None)
    if user_trainlog_cfg and hasattr(user_trainlog_cfg, 'num_val_ep'):
        total_val_ep = user_trainlog_cfg.num_val_ep
        logger.info(f"✅ OVERRIDE: num_val_ep = {total_val_ep} from user config")
    else:
        logger.info(f"ℹ️ Using num_val_ep = {total_val_ep} from training config")

    # 2. Try to get max_episodes_per_symbol from GLOBAL scope of user_cfg_module
    #    because it is not part of the Pydantic model "cfg"
    per_sym_ep = 5000 # Default fallback
    if hasattr(user_cfg_module, 'max_episodes_per_symbol'):
        per_sym_ep = getattr(user_cfg_module, 'max_episodes_per_symbol')
        logger.info(f"✅ Found global 'max_episodes_per_symbol' = {per_sym_ep}")
    else:
        logger.info(f"ℹ️ 'max_episodes_per_symbol' not found in config, using default: {per_sym_ep}")

    
    sequences, keys = create_validation_episodes(
        val_sequences=sequences, val_keys=keys,
        num_episodes=total_val_ep,
        max_episodes_per_symbol=per_sym_ep,
        seed=cfg.get("random_seed", 404)
    )

    seq_cfg = cfg.get("seq", {})
    data_cfg = cfg.get("data", {})
    market_cfg = cfg.get("market", {})
    
    if 'backtest' not in cfg: cfg['backtest'] = {}

    if user_cfg_obj is not None:
        logger.info(f"📂 Reading user config from {args.config} (Exec method)...")
        try:
            u_backtest = getattr(user_cfg_obj, 'backtest', None)
            if isinstance(u_backtest, dict):
                ovr_risk = u_backtest.get('use_risk_management')
                ovr_tsl = u_backtest.get('trailing_stop')
                ovr_tsl_min = u_backtest.get('trailing_stop_min')
                ovr_fee = u_backtest.get('fee_buffer_mult')
                ovr_hyst = u_backtest.get('delta_p_hysteresis')
            elif u_backtest is not None:
                ovr_risk = getattr(u_backtest, 'use_risk_management', None)
                ovr_tsl = getattr(u_backtest, 'trailing_stop', None)
                ovr_tsl_min = getattr(u_backtest, 'trailing_stop_min', None)
                ovr_fee = getattr(u_backtest, 'fee_buffer_mult', None)
                ovr_hyst = getattr(u_backtest, 'delta_p_hysteresis', None)
            else:
                ovr_risk, ovr_tsl = None, None
                logger.warning("⚠️ 'backtest' not found in user_cfg object")

            if ovr_risk is not None:
                cfg['backtest']['use_risk_management'] = bool(ovr_risk)
                logger.info(f"✅ OVERRIDE: use_risk_management = {bool(ovr_risk)}")
            if ovr_tsl is not None:
                cfg['backtest']['trailing_stop'] = float(ovr_tsl)
                logger.info(f"✅ OVERRIDE: trailing_stop = {float(ovr_tsl)}")
            if ovr_tsl_min is not None: cfg['backtest']['trailing_stop_min'] = float(ovr_tsl_min)
            if ovr_fee is not None: cfg['backtest']['fee_buffer_mult'] = float(ovr_fee)
            if ovr_hyst is not None: cfg['backtest']['delta_p_hysteresis'] = float(ovr_hyst)
        except Exception as e:
            logger.error(f"❌ Error extracting config: {e}")
    else:
        logger.error("❌ Could not find 'cfg' variable in the config file!")

    backtest_cfg = cfg.get("backtest", {})
    use_risk_mgmt = backtest_cfg.get("use_risk_management", False)
    tsl_stop = backtest_cfg.get("trailing_stop", 0.018)
    tsl_min = backtest_cfg.get("trailing_stop_min", 0.005)
    fee_buf_mult = backtest_cfg.get("fee_buffer_mult", 2.5)
    delta_hyst = backtest_cfg.get("delta_p_hysteresis", 0.0015)
    
    logger.warning("=" * 80)
    logger.warning(f"🎯 FINAL TSL PARAMS TO BE USED:")
    logger.warning(f"   use_risk_mgmt = {use_risk_mgmt}")
    logger.warning(f"   tsl_stop      = {tsl_stop}")
    logger.warning(f"   tsl_min       = {tsl_min}")
    logger.warning("=" * 80)
    
    logger.info(f"🛡️ Risk Management: {use_risk_mgmt}")
    if use_risk_mgmt:
        logger.info(f"   TSL: {tsl_stop*100:.2f}% -> {tsl_min*100:.2f}%")
        logger.info(f"   Fee Buffer: {fee_buf_mult}x, Delta Hysteresis: {delta_hyst*100:.3f}%")

    model_cfg = cfg.get("model", {})
    rl_cfg = cfg.get("rl", {})
    per_cfg = cfg.get("per", {})
    eps_cfg = cfg.get("eps", {})
    
    ensemble_cfg = ensemble_cfg_obj
    
    long_threshold_val = 0.02
    if args.long_threshold is not None: long_threshold_val = args.long_threshold
    elif ensemble_cfg and hasattr(ensemble_cfg, 'long_threshold'): long_threshold_val = ensemble_cfg.long_threshold
    elif ensemble_cfg and hasattr(ensemble_cfg, 'threshold'): long_threshold_val = ensemble_cfg.threshold
        
    short_threshold_val = 0.02
    if args.short_threshold is not None: short_threshold_val = args.short_threshold
    elif ensemble_cfg and hasattr(ensemble_cfg, 'short_threshold'): short_threshold_val = ensemble_cfg.short_threshold
    elif ensemble_cfg and hasattr(ensemble_cfg, 'threshold'): short_threshold_val = ensemble_cfg.threshold

    disable_cross_close = False
    conflict_cooldown_bars = 0
    if ensemble_cfg and hasattr(ensemble_cfg, 'disable_cross_close'):
        disable_cross_close = ensemble_cfg.disable_cross_close
        if disable_cross_close: logger.info("ℹ️ Cross-closing logic is DISABLED by config.")

    confidence_ratio = 1.2
    if ensemble_cfg and hasattr(ensemble_cfg, 'confidence_ratio'):
        confidence_ratio = ensemble_cfg.confidence_ratio
        logger.info(f"ℹ️ Soft conflict resolution is ENABLED with confidence ratio: {confidence_ratio}")
    
    num_channels = cfg.get("num_channels", 10)
    default_datachannels = ['open', 'high', 'low', 'close', 'volume']
    if num_channels > 5: default_datachannels += [f"feat_{i}" for i in range(5, num_channels)]

    env_num_actions = market_cfg.get("num_actions", 3)

    env_params = {
        "sequences": sequences, "stats": all_stats, "keys": keys, "render_mode": None,
        "full_seq_len": seq_cfg.get("full_seq_len", 150),
        "num_features": num_channels,
        "flat_state_size": 0,
        "initial_balance": market_cfg.get("initial_balance", 10000.0),
        "pre_signal_len": seq_cfg.get("pre_signal_len", 90),
        "datachannels": data_cfg.get("datachannels", default_datachannels),
        "agent_session_len": seq_cfg.get("agent_session_len", 60),
        "agent_history_len": seq_cfg.get("agent_history_len", 90),
        "input_history_len": seq_cfg.get("input_history_len", 90),
        "pricechannels": [0, 1, 2, 3],
        "volumechannels": [4],
        "otherchannels": list(range(5, num_channels)),
        "action_history_len": seq_cfg.get("action_history_len", 2),
        "inaction_penalty_ratio": market_cfg.get("inaction_penalty_ratio", 0.0),
        "backtest_mode": True,
        "num_actions": env_num_actions,
        "allowed_directions": market_cfg.get("allowed_directions", ['LONG', 'SHORT']),
        "filter_direction": None,
        "transaction_fee": market_cfg.get("transaction_fee", 0.0004),
        "slippage": market_cfg.get("slippage", 0.0002),
        "position_fraction": market_cfg.get("position_fraction", 0.1),
        "use_risk_management": use_risk_mgmt
    }
    
    env_long, env_short, env = None, None, None
    logger.info("🌍 Initializing TradingEnvironment(s)...")

    if args.ensemble:
        env_params_long = env_params.copy()
        env_params_long["allowed_directions"] = ['LONG']
        env_params_long["num_actions"] = 4 
        env_long = TradingEnvironment(**env_params_long)
        
        env_params_short = env_params.copy()
        env_params_short["allowed_directions"] = ['SHORT']
        env_params_short["num_actions"] = 4 
        env_short = TradingEnvironment(**env_params_short)
        
        assert len(env_long.sequences) == len(env_short.sequences), "Sequence count mismatch"
        logger.info(" -> LONG and SHORT environments created for ensemble.")
    else:
        env = TradingEnvironment(**env_params)
        logger.info(" -> Single environment created.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Using device: {device}")

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
            device=device,
            gamma=rl_cfg.get("gamma", 0.99),
            learning_rate=rl_cfg.get("lr", 1e-4),
            batch_size=rl_cfg.get("batch_size", 32),
            buffer_size=100,
            perf_cfg=PerformanceConfig(),
            dropout_model=model_cfg.get("dropout_p", 0.0),
            target_update_freq=rl_cfg.get("target_update_freq", 1000),
            train_start=rl_cfg.get("train_start", 1000),
            per_alpha=per_cfg.get("alpha", 0.6),
            per_beta_start=per_cfg.get("beta_start", 0.4),
            per_beta_frames=per_cfg.get("beta_frames", 10000),
            eps_start=eps_cfg.get("eps_start", 1.0),
            eps_end=eps_cfg.get("eps_end", 0.01),
            eps_frames=eps_cfg.get("eps_frames", 10000),
            epsilon=eps_cfg.get("eps_start", 1.0),
            max_gradient_norm=rl_cfg.get("max_gradient_norm", 1.0)
        )

    agent = None
    if args.ensemble:
        ensemble_cfg = ensemble_cfg_obj
        use_conf = getattr(ensemble_cfg, 'use_confidence', False) if ensemble_cfg else False
        enable_long = getattr(ensemble_cfg, 'enable_long', True) if ensemble_cfg else True
        enable_short = getattr(ensemble_cfg, 'enable_short', True) if ensemble_cfg else True
        
        agent = EnsembleAgent(
            long_agent_path=args.long_model,
            short_agent_path=args.short_model,
            device=device,
            agent_creator=create_agent,
            use_confidence=use_conf,
            long_threshold=long_threshold_val,
            short_threshold=short_threshold_val,
            enable_long=enable_long,
            enable_short=enable_short,
            verbose=args.ensemble_verbose
        )
    else:
        print(f"\n📦 Loading single agent model from: {model_path}")
        def get_specialist_action_dim(model_path): return 3
        num_actions_env = 3
        if "SHORT_ONLY" in model_path or "LONG_ONLY" in model_path: num_actions_env = 3
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
    
    pbar = tqdm(range(len(sequences)), desc="Simulating")
    
    for i in pbar:
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

        episode_bars = 0
        if args.ensemble:
            if i >= len(env_long.sequences) or i >= len(env_short.sequences): continue

            obs_l, info_l = env_long.reset(options={"forced_index": i})
            obs_s, info_s = env_short.reset(options={"forced_index": i})
            
            done_l, done_s = False, False
            loop_safety_counter = 0
            MAX_LOOP_STEPS = 500

            while not (done_l and done_s):
                loop_safety_counter += 1
                if loop_safety_counter > MAX_LOOP_STEPS: break

                if not done_l: current_step = env_long.step_idx
                elif not done_s: current_step = env_short.step_idx
                else: break
                
                # Update episode bars count
                episode_bars = max(episode_bars, current_step)
                
                # --- УЛУЧШЕННАЯ ЛОГИКА АНСАМБЛЯ ---
                final_act_l, final_act_s = 0, 0

                # 1. Получаем сигналы от специалистов
                long_wants_open, long_conf = (agent.get_long_vote(obs_l) if not done_l else (False, 0.0))
                short_wants_open, short_conf = (agent.get_short_vote(obs_s) if not done_s else (False, 0.0))

                # 2. Определяем текущее состояние позиций
                long_is_active = (env_long.position > 0)
                short_is_active = (env_short.position < 0)

                # 3. Обнаруживаем конфликт для разрешения
                is_conflict = ((long_wants_open and short_is_active) or
                               (short_wants_open and long_is_active) or
                               (long_wants_open and short_wants_open))

                if is_conflict:
                    # --- SOFT CONFLICT RESOLUTION (No future ban) ---
                    if long_conf > (short_conf * confidence_ratio):
                        short_wants_open = False # Long wins, Short is ignored for this tick
                        logger.info(f"Conflict on {ticker_name}: LONG wins ({long_conf:.4f} vs {short_conf:.4f}). Ignoring SHORT.")
                    elif short_conf > (long_conf * confidence_ratio):
                        long_wants_open = False # Short wins, Long is ignored
                        logger.info(f"Conflict on {ticker_name}: SHORT wins ({short_conf:.4f} vs {long_conf:.4f}). Ignoring LONG.")
                    else: # Uncertainty is too high, both HOLD
                        long_wants_open = False
                        short_wants_open = False
                        logger.info(f"Conflict on {ticker_name}: Too close to call ({long_conf:.4f} vs {short_conf:.4f}). Both HOLD.")

                # 4. Применяем логику с УЖЕ разрешенным конфликтом
                if disable_cross_close:
                        # Запрет перекрестного закрытия: сигнал на открытие игнорируется, если активна противоположная позиция
                        if long_wants_open and short_is_active:
                            pass # Игнорируем LONG сигнал
                        elif short_wants_open and long_is_active:
                            pass # Игнорируем SHORT сигнал
                        else:
                            # Разрешаем открытие, только если нет конфликта позиций
                            if long_wants_open and not long_is_active and long_conf > agent.long_threshold:
                                final_act_l = 1
                            if short_wants_open and not short_is_active and short_conf > agent.short_threshold:
                                final_act_s = 2
                else:
                    # Перекрестное закрытие разрешено: один агент может закрыть позицию другого
                    if long_wants_open and short_is_active:
                        final_act_s = 3  # Закрыть SHORT
                        if long_conf > agent.long_threshold: final_act_l = 1 # Открыть LONG
                    elif short_wants_open and long_is_active:
                        final_act_l = 3  # Закрыть LONG
                        if short_conf > agent.short_threshold: final_act_s = 2 # Открыть SHORT
                    else:
                        # Если нет активных позиций, открываемся по сигналу
                        if long_wants_open and long_conf > agent.long_threshold:
                            final_act_l = 1
                        if short_wants_open and short_conf > agent.short_threshold:
                            final_act_s = 2

                if not done_l:
                    next_obs_l, _, term_l, trunc_l, info_l = env_long.backtest_step(
                        action=final_act_l, signal_dt=signal_dt, ticker=ticker_name,
                        trailing_stop=tsl_stop, trailing_stop_min=tsl_min,
                        fee_buffer_mult=fee_buf_mult, delta_p_hysteresis=delta_hyst
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

                if not done_s:
                    next_obs_s, _, term_s, trunc_s, info_s = env_short.backtest_step(
                        action=final_act_s, signal_dt=signal_dt, ticker=ticker_name,
                        trailing_stop=tsl_stop, trailing_stop_min=tsl_min,
                        fee_buffer_mult=fee_buf_mult, delta_p_hysteresis=delta_hyst
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
            
            # --- FIX: Accumulate actual bars processed ---
            total_bars_processed += episode_bars
            
        else:
            obs, _ = env.reset(options={"forced_index": i})
            done = False
            episode_bars = 0
            while not done:
                action = agent.select_action(obs, training=False)
                next_obs, reward, terminated, truncated, info = env.backtest_step(
                    action=action, signal_dt=signal_dt, ticker=ticker_name
                )
                episode_bars += 1
                if info.get('position_closed'):
                    pnl = info["trade_realized_pnl"]
                    comm = info.get("trade_commission", 0.0)
                    net_pnl = pnl - comm
                    trade_data = {
                        "symbol": ticker_name,
                        "direction": info.get("direction", "UNKNOWN"),
                        "pnl": pnl, "net_pnl": net_pnl, "commission": comm,
                        "bars": info.get("holding_duration_bars", 0),
                        "tsl_triggered": info.get('tsl_triggered', False)
                    }
                    all_trades.append(trade_data)
                obs = next_obs
                done = terminated or truncated
            
            # --- FIX: Accumulate actual bars processed ---
            total_bars_processed += episode_bars

        pbar.set_postfix({
            "PnL": f"{sum(t.get('net_pnl', 0.0) for t in all_trades):,.0f}",
            "Trds": len(all_trades)
        })
    
    logging.getLogger().setLevel(logging.INFO)
    
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
    
    # --- FIX: Correct trading days calculation ---
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
        
    # --- Advanced Exit Stats ---
    tsl_win = 0
    tsl_loss = 0
    time_win = 0
    time_loss = 0
    
    for t in all_trades:
        is_win = t.get('net_pnl', 0.0) > 0
        triggered_tsl = t.get('tsl_triggered', False)
        
        if triggered_tsl:
            if is_win: tsl_win += 1
            else: tsl_loss += 1
        else:
            if is_win: time_win += 1
            else: time_loss += 1
            
    tsl_total = tsl_win + tsl_loss
    time_total = time_win + time_loss

    print("\n" + "="*44)
    print("📊 FINAL VALIDATION RESULTS")
    print("="*44)
    
    if args.ensemble:
        print(f"Mode: ENSEMBLE (LONG + SHORT specialists) | Long Th: {long_threshold_val} | Short Th: {short_threshold_val}")
    else:
        print(f"Mode: Single agent")
        
    print(f"Trades: {total_trades} (Long: {long_trades}, Short: {short_trades}, Win: {win_count}, Loss: {loss_count}) | WinRate: {wr_ratio:.2%} | PF: {profit_factor:.4f}")
    print(f"Gross PnL: {gross_pnl:.2f} | Net PnL: {net_pnl:.2f} | Commission: {total_commission:.2f} | Avg/Trade: {avg_pnl_per_trade:.2f}")
    print(f"Best Trade: {best_trade:+.2f} | Worst Trade: {worst_trade:+.2f} | MaxDD: {abs(max_dd):.2%} | Sharpe (Per Trade): {sharpe:.3f} | Sortino (Per Trade): {sortino:.3f}")
    print(f"Avg Hold: {avg_holding_time:.2f} bars | Min Hold: {min_holding_time} bars | Max Hold: {max_holding_time} bars")
    print(f"Duration: {total_duration:.2f}s | Bars: {total_bars_processed} | Trading Days: {trading_time_days:.1f}")
    print(f"PnL/Day: {pnl_per_day:.2f} USDT | ROI: {roi_percent:.2f}% | Annualized ROI: {roi_annualized:.1f}%")
    print(f"Commission: {(total_commission / max(1e-9, abs(gross_pnl)))*100:.1f}% of gross | Avg Win: {avg_win_size:.2f} | Avg Loss: {avg_loss_size:.2f} | W/L Ratio: {win_loss_ratio:.2f}")
    print(f"Expectancy/Trade: {expectancy:.2f} USDT")
    
    print("-" * 44)
    print("🛑 Exit Analysis:")
    print(f"  TSL (Take Profit/Trail): {tsl_win} ({tsl_win/max(1, total_trades):.1%}) - \"TSL\"")
    print(f"  TSL SL (Stop Loss):      {tsl_loss} ({tsl_loss/max(1, total_trades):.1%}) - \"TSL SL\"")
    print(f"  Time Win (Timeout):      {time_win} ({time_win/max(1, total_trades):.1%}) - \"Time\" (Profit)")
    print(f"  Time Loss (Timeout):     {time_loss} ({time_loss/max(1, total_trades):.1%}) - \"Time SL\" (Loss)")
    print(f"  Total TSL Hits: {tsl_total}")
    print("="*44)

if __name__ == "__main__":
    run_validation()
