# validate_model.py
import argparse
import json
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
import datetime as dt

# --- Project Imports ---
# Ensure this script is run from the project root or that the path is configured correctly.
try:
    from trading_environment import TradingEnvironment
    from agent import D3QN_PER_Agent
    from config import PerformanceConfig
except ImportError:
    print("ERROR: Make sure to run this script from the project root, e.g., `python third_party/rl-trading-binance/validate_model.py ...`")
    exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def load_data(data_path: str, norm_stats_path: str):
    """Loads and normalizes data exactly as it was done for training."""
    logger.info(f"Loading normalization stats from {norm_stats_path}")
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")
    with open(norm_stats_path, 'r') as f:
        stats = json.load(f)
    
    means = np.array(stats['mean'], dtype=np.float32)
    stds = np.array(stats['std'], dtype=np.float32) + 1e-8

    logger.info(f"Loading validation data from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation data file not found: {data_path}")
        
    sequences = []
    keys = []
    with np.load(data_path, allow_pickle=True) as d:
        keys = [k for k in d.files if not k.startswith('_')]
        for key in tqdm(keys, desc="Loading and normalizing data"):
            raw_seq = d[key].astype(np.float32)
            norm_seq = (raw_seq - means) / stds
            # Reshape for CNN: (L, C) -> (C, L, 1)
            norm_seq = norm_seq.T
            norm_seq = np.expand_dims(norm_seq, -1)
            sequences.append(norm_seq)
            
    return sequences, stats, keys

def calculate_and_log_metrics(trade_pnls: list, initial_balance: float, episodes: int, label="Validation"):
    """Calculates and logs key performance metrics."""
    if not trade_pnls:
        logger.warning("No trades were made, cannot calculate performance metrics.")
        return

    returns = np.array(trade_pnls) / initial_balance
    
    # --- Sharpe & Sortino ---
    mean_r = returns.mean()
    std_r = returns.std(ddof=1) if len(returns) > 1 else 0
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-9 else 0 # Annualized Sharpe

    downside_returns = returns[returns < 0]
    downside_dev = np.sqrt((downside_returns**2).mean()) if len(downside_returns) > 0 else 0
    sortino = (mean_r / downside_dev) * np.sqrt(252) if downside_dev > 1e-9 else (float("inf") if mean_r > 0 else 0)

    # --- Profit Factor & Win Rate ---
    pos_pnls = [p for p in trade_pnls if p > 0]
    neg_pnls = [p for p in trade_pnls if p < 0]
    profit_factor = sum(pos_pnls) / abs(sum(neg_pnls)) if sum(neg_pnls) != 0 else float("inf")
    win_rate = len(pos_pnls) / len(trade_pnls) if trade_pnls else 0

    # --- Max Drawdown ---
    equity_curve = np.cumsum(trade_pnls) + initial_balance
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # --- Logging ---
    logger.info("="*50)
    logger.info(f"📊 {label} Results ({episodes} episodes)")
    logger.info("="*50)
    logger.info(f"Total PnL:              ${sum(trade_pnls):,.2f}")
    logger.info(f"Total Trades:           {len(trade_pnls)}")
    logger.info(f"Win Rate:               {win_rate:.2%}")
    logger.info(f"Profit Factor:          {profit_factor:.3f}")
    logger.info(f"Max Drawdown:           {max_dd:.2%}")
    logger.info(f"Annualized Sharpe Ratio:  {sharpe:.3f}")
    logger.info(f"Annualized Sortino Ratio: {sortino:.3f}")
    logger.info("="*50)

def run_validation(config_path: str, model_path: str):
    """Main validation function."""
    logger.info(f"Starting validation for model: {model_path}")
    
    # 1. Load Configuration from JSON
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # 2. Setup Environment
    seed = cfg.get('random_seed', 404)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        if cfg.get('deterministic', False) or cfg.get('perf', {}).get('cudnn_deterministic', False):
            logger.info("Enabling full deterministic mode for CUDA.")
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    # 3. Load Data
    # The paths in the JSON are relative to the 'third_party/rl-trading-binance' directory.
    subproject_root = 'third_party/rl-trading-binance'
    val_data_path = os.path.join(subproject_root, cfg['paths']['val_data_path'])
    norm_stats_path = os.path.join(os.path.dirname(model_path), 'norm_stats.json')
    
    sequences, stats_dict, keys = load_data(val_data_path, norm_stats_path)
    
    num_episodes = min(cfg['trainlog']['num_val_ep'], len(sequences))
    logger.info(f"Running validation for {num_episodes} episodes.")
    sequences = sequences[:num_episodes]
    keys = keys[:num_episodes]

    # 4. Initialize Environment
    # Calculate flat_state_size based on config
    num_features = cfg.get('num_channels', cfg.get('data', {}).get('num_channels'))
    input_history_len = cfg['seq'].get('input_history_len', cfg['seq']['agent_history_len'])
    action_history_len = cfg['seq']['action_history_len']
    num_actions = cfg['market']['num_actions']
    extras = 4  # position, unrealized, time_elapsed, time_remaining
    history_vector_size = num_actions * action_history_len if action_history_len > 0 else 0
    flat_features = input_history_len * num_features
    flat_state_size = flat_features + extras + history_vector_size

    env_kwargs = {
        "sequences": sequences,
        "stats": stats_dict,
        "initial_balance": cfg['market']['initial_balance'],
        "num_actions": cfg['market']['num_actions'],
        "agent_history_len": cfg['seq']['agent_history_len'],
        "agent_session_len": cfg['seq']['agent_session_len'],
        "backtest_mode": True,
        "use_risk_management": cfg['backtest'].get('use_risk_management', True),
        "render_mode": cfg.get('render_mode'),
        "full_seq_len": cfg['seq']['full_seq_len'],
        "num_features": num_features,
        "flat_state_size": flat_state_size,
        "pre_signal_len": cfg['seq']['pre_signal_len'],
        "data_channels": cfg['data']['data_channels'],
        "slippage": cfg['market']['slippage'],
        "transaction_fee": cfg['market']['transaction_fee'],
        "input_history_len": input_history_len,
        "price_channels": cfg['data']['price_channels'],
        "volume_channels": cfg['data']['volume_channels'],
        "other_channels": cfg['data']['other_channels'],
        "action_history_len": action_history_len,
        "inaction_penalty_ratio": cfg['market']['inaction_penalty_ratio'],
    }
    env = TradingEnvironment(**env_kwargs)

    # 5. Initialize Agent
    perf_cfg = PerformanceConfig()
    perf_cfg.use_amp = cfg['perf']['use_amp']
    
    # Filter out 'name' if it exists in the dictionaries
    model_params = {k: v for k, v in cfg.get('model', {}).items() if k != 'name'}
    rl_params = {k: v for k, v in cfg.get('rl', {}).items() if k != 'name'}
    per_params = {k: v for k, v in cfg.get('per', {}).items() if k != 'name'}
    eps_params = {k: v for k, v in cfg.get('eps', {}).items() if k != 'name'}

    # Handle parameter name mismatches between config and agent constructor
    if 'dropout_p' in model_params:
        model_params['dropout_model'] = model_params.pop('dropout_p')
    if 'lr' in rl_params:
        rl_params['learning_rate'] = rl_params.pop('lr')
    rl_params.pop('clip_range', None)
    rl_params.pop('n_step', None)
    rl_params.pop('gamma_n_step_buffer', None)
    if 'per_eps' in per_params:
        per_params['epsilon'] = per_params.pop('per_eps')
    if 'eps_decay_frames' in eps_params:
        eps_params['eps_frames'] = eps_params.pop('eps_decay_frames')

    agent = D3QN_PER_Agent(
        state_shape=tuple(cfg['state_shape']),
        action_dim=cfg['market']['num_actions'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        perf_cfg=perf_cfg,
        **model_params,
        **rl_params,
        **per_params,
        **eps_params
    )
    agent.load_model(model_path, strict=True)
    agent.policy_net.eval()

    # 6. Run Validation Loop
    all_trade_pnls = []
    stub_dt = dt.datetime(2000, 1, 1, 0, 0)
    
    for i in tqdm(range(num_episodes), desc="Validating Episodes"):
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        
        try:
            ticker_name = keys[i].split('_')[0]
        except (IndexError, AttributeError):
            ticker_name = "UNKNOWN"

        while not done:
            action = agent.select_action(obs, training=False)
            
            # Per user request: TSL only, no separate SL/TP
            backtest_kwargs = {
                'stop_loss': None,
                'take_profit': None,
                'trailing_stop': cfg['backtest'].get('trailing_stop'),
                'trailing_stop_min': cfg['backtest'].get('trailing_stop_min'),
                'fee_buffer_mult': cfg['backtest'].get('fee_buffer_mult'),
                'delta_p_hysteresis': cfg['backtest'].get('delta_p_hysteresis'),
            }
            
            obs, _, terminated, truncated, info = env.backtest_step(
                action=action,
                signal_dt=stub_dt,
                ticker=ticker_name,
                **backtest_kwargs
            )
            done = terminated or truncated
            
            if info.get("position_closed", False):
                pnl = info.get("trade_realized_pnl", 0.0)
                all_trade_pnls.append(pnl)

    # 7. Calculate and Log Metrics
    calculate_and_log_metrics(all_trade_pnls, cfg['market']['initial_balance'], num_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation for a trained model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config_train.json file from a training run."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the best.pth or final.pth model file."
    )
    args = parser.parse_args()
    
    # The arguments are passed as paths relative to the project root where the command is run.
    run_validation(config_path=args.config, model_path=args.model)