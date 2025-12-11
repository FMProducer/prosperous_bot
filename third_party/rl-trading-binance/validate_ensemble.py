#
# D3QN Validation Script with Ensemble Support
#
import argparse
import json
import logging
import time
import datetime
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os

# Assuming these are the correct paths from the project structure
from trading_environment import TradingEnvironment
from agent import D3QN_PER_Agent as D3QNPERAgent

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

class EnsembleAgent:
    """
    Tier 1 Ensemble: Direction Agreement + Threshold
    Combines a Long-Only Specialist and a Short-Only Specialist.
    """
    def __init__(self, long_model_path, short_model_path, state_shape, device, threshold=0.02, agent_params=None):
        self.device = device
        self.threshold = threshold
        self.long_model_path = long_model_path
        self.short_model_path = short_model_path
        
        # Initialize Specialists
        # CRITICAL: Specialists have action_dim=3 (HOLD, OPEN, CLOSE)
        specialist_action_dim = 3
        
        # Common params for agents
        ap = agent_params if agent_params else {}
        
        logger.info(f"🤖 Initializing Ensemble Long Specialist from {long_model_path}")
        self.long_agent = D3QNPERAgent(
            state_shape=state_shape, 
            action_dim=specialist_action_dim, 
            device=device,
            **ap
        )
        self.long_agent.load_model(long_model_path)
        self.long_agent.policy_net.eval()
        
        logger.info(f"🤖 Initializing Ensemble Short Specialist from {short_model_path}")
        self.short_agent = D3QNPERAgent(
            state_shape=state_shape, 
            action_dim=specialist_action_dim, 
            device=device,
            **ap
        )
        self.short_agent.load_model(short_model_path)
        self.short_agent.policy_net.eval()

    def select_action(self, obs, training=False, position=0):
        # obs shape: (1, C, L) usually
        with torch.no_grad():
            # Get Q-values from both specialists [HOLD, OPEN, CLOSE]
            q_long = self.long_agent.get_q_values(obs) 
            q_short = self.short_agent.get_q_values(obs)
            
        # --- Tier 1 Logic ---
        action = 0 # Default HOLD
        
        # Calculate Confidence: Q_OPEN - Q_HOLD
        # Index 1 is OPEN for both specialists
        conf_long = q_long[1] - q_long[0]
        conf_short = q_short[1] - q_short[0]
        
        if position == 0: # FLAT
            want_long = conf_long > self.threshold
            want_short = conf_short > self.threshold
            
            if want_long and not want_short:
                action = 1 # ENV: LONG
            elif want_short and not want_long:
                action = 2 # ENV: SHORT
            elif want_long and want_short:
                # Conflict resolution: Pick the stronger signal
                action = 1 if conf_long > conf_short else 2
                
        elif position > 0: # LONG
            # Check Long Agent for Close (Index 2)
            if q_long[2] > q_long[0]:
                action = 3 # ENV: CLOSE
                
        elif position < 0: # SHORT
            # Check Short Agent for Close (Index 2)
            if q_short[2] > q_short[0]:
                action = 3 # ENV: CLOSE
                
        return action

class PerformanceConfig:
    def __init__(self):
        self.USE_CUDA_IF_AVAILABLE = True
        self.TRAIN_DEVICE = "cuda"
        self.REPLAY_DEVICE = "cpu"

def main():
    parser = argparse.ArgumentParser(description="Ensemble and Single Agent Backtest Validation")
    
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the JSON config file.')
    parser.add_argument('--npz-file', type=str, required=True, help='Path to the .npz file with sequences and stats.')
    
    # For single agent mode
    parser.add_argument('-m', '--model', type=str, help='Path to the model checkpoint for a single agent.')
    
    # For ensemble mode
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble mode.')
    parser.add_argument('--long-model', type=str, help='Path to the LONG specialist model.')
    parser.add_argument('--short-model', type=str, help='Path to the SHORT specialist model.')
    
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.ensemble:
        if not args.long_model or not args.short_model:
            parser.error("--long-model and --short-model are required when using --ensemble.")
        if args.model:
            parser.error("-m/--model cannot be used with --ensemble.")
    else:
        if not args.model:
            parser.error("-m/--model is required when not using --ensemble.")
        if args.long_model or args.short_model:
            parser.error("--long-model/--short-model can only be used with --ensemble.")

    # --- Load Config and Data ---
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        
    model_path = args.model if not args.ensemble else None

    logger.info(f"💾 Loading data from {args.npz_file}...")
    data = np.load(args.npz_file, allow_pickle=True)
    sequences = data['sequences']
    all_stats = data['stats'].item()
    keys = data['keys']
    
    # --- Config Extraction ---
    train_cfg_dict = cfg.get("training_config", {})
    env_params = cfg.get("environment_parameters", {})
    model_cfg = cfg.get("model_config", {})
    rl_cfg = cfg.get("rl_config", {})
    backtest_kwargs = cfg.get("backtest_kwargs", {})
    per_cfg = cfg.get("per_config", {})
    ensemble_cfg = cfg.get("ensemble_config", {}) # Читаем из JSON (после dump) или объекта
    eps_cfg = cfg.get("eps", {})
    
    # Override Environment parameters for Ensemble Mode
    if args.ensemble:
        logger.info("🎭 Configuring Environment for Ensemble Mode (LONG + SHORT)")
        env_params['num_actions'] = 4 # HOLD, LONG, SHORT, CLOSE
        env_params['allowed_directions'] = ['LONG', 'SHORT']
        env_params['filter_direction'] = None # Ensure we don't filter sequences
    
    # Initialize Environment
    logger.info("🌍 Initializing TradingEnvironment...")
    env = TradingEnvironment(
        sequences=sequences, 
        stats=all_stats, 
        keys=keys, 
        render_mode=None, 
        **env_params
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = None
    if args.ensemble:
        # Extract Agent Params from config to reuse architecture settings
        agent_params = {
            'cnn_maps': model_cfg.get('cnn_maps'),
            'cnn_kernels': model_cfg.get('cnn_kernels'),
            'cnn_strides': model_cfg.get('cnn_strides'),
            'cnn_dilations': model_cfg.get('cnn_dilations'),
            'dense_val': model_cfg.get('dense_val'),
            'dense_adv': model_cfg.get('dense_adv'),
            'dropout_p': model_cfg.get('dropout_p', 0.0),
            'gamma': rl_cfg.get('gamma', 0.99),
            'learning_rate': rl_cfg.get('lr', 1e-4),
            'batch_size': rl_cfg.get('batch_size', 32),
            'buffer_size': 100, # Dummy for validation
            'perf_cfg': PerformanceConfig()
        }
        
        agent = EnsembleAgent(
            long_model_path=args.long_model,
            short_model_path=args.short_model,
            state_shape=train_cfg_dict.get('state_shape', (10, 90, 1)),
            device=device,
            threshold=ensemble_cfg.get('threshold', 0.02), # Load from config or default
            agent_params=agent_params
        )
    else:
        # Standard Single Agent Initialization
        agent = D3QNPERAgent(
            state_shape=train_cfg_dict.get('state_shape', (10, 90, 1)),
            action_dim=env_params['num_actions'],
            cnn_maps=model_cfg.get('cnn_maps'),
            cnn_kernels=model_cfg.get('cnn_kernels'),
            cnn_strides=model_cfg.get('cnn_strides'),
            cnn_dilations=model_cfg.get('cnn_dilations'),
            dense_val=model_cfg.get('dense_val'),
            dense_adv=model_cfg.get('dense_adv'),
            additional_feats=model_cfg.get('additional_feats', 12),
            dropout_p=model_cfg.get('dropout_p', 0.0),
            device=device,
            gamma=rl_cfg.get('gamma', 0.99),
            learning_rate=rl_cfg.get('lr', 1e-4),
            batch_size=rl_cfg.get('batch_size', 32),
            buffer_size=per_cfg.get('buffer_size', 100000),
            perf_cfg=PerformanceConfig()
        )
        logger.info(f"🧠 Initializing Single Agent from {model_path}...")
        agent.load_model(model_path)

    logger.info("🚀 Starting Backtest Validation...")
    
    # --- Validation Loop ---
    all_trades_info = []
    total_bars_processed = 0
    start_time = time.time()
    
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    
    pbar = tqdm(range(len(sequences)), desc="Simulating")
    
    for i in pbar:
        obs, _ = env.reset(options={"forced_index": i})
        done = False
        
        # Date/Ticker parsing for logging
        signal_dt_for_step = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        ticker_name = "UNKNOWN"
        if keys and i < len(keys):
            try:
                key_parts = keys[i].split('_')
                ticker_name = key_parts[0]
                if len(key_parts) > 1:
                    start_dt_str = key_parts[1]
                    signal_dt_for_step = datetime.datetime.fromisoformat(start_dt_str.replace('Z', '+00:00'))
            except Exception:
                pass

        while not done:
            # Pass position to select_action if ensemble, else just obs
            if args.ensemble:
                action = agent.select_action(obs, training=False, position=env.position)
            else:
                action = agent.select_action(obs, training=False)
            
            next_obs, reward, terminated, truncated, info = env.backtest_step(
                action=action,
                signal_dt=signal_dt_for_step,
                ticker=ticker_name,
                **backtest_kwargs
            )
            
            done = terminated or truncated
            obs = next_obs
            total_bars_processed += 1
            
            if info.get('position_closed', False):
                all_trades_info.append(info)
                
        pbar.set_postfix({
            'PnL': f"{sum(t.get('trade_realized_pnl', 0.0) for t in all_trades_info):.0f}", 
            'Trds': len(all_trades_info)
        })

    # --- Metrics Calculation & Printing ---
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"\n--- Backtest Summary ---")
    logger.info(f"Total simulation time: {duration:.2f} seconds")
    logger.info(f"Total bars processed: {total_bars_processed}")

    if not all_trades_info:
        logger.warning("No trades were executed. Cannot calculate performance metrics.")
    else:
        trades_df = pd.DataFrame(all_trades_info)
        
        # --- Calculate and Print Metrics ---
        num_trades = len(trades_df)
        pnl = trades_df['trade_realized_pnl'].sum()
        
        win_trades = trades_df[trades_df['trade_realized_pnl'] > 0]
        loss_trades = trades_df[trades_df['trade_realized_pnl'] < 0]
        
        num_wins = len(win_trades)
        num_losses = len(loss_trades)
        win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
        
        avg_win = win_trades['trade_realized_pnl'].mean() if num_wins > 0 else 0
        avg_loss = loss_trades['trade_realized_pnl'].mean() if num_losses > 0 else 0
        
        profit_factor = win_trades['trade_realized_pnl'].sum() / abs(loss_trades['trade_realized_pnl'].sum()) if num_losses > 0 else float('inf')
        
        avg_trade_pnl = trades_df['trade_realized_pnl'].mean()
        
        logger.info(f"Total Trades: {num_trades}")
        logger.info(f"Total PnL: ${pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.2f}% ({num_wins} wins / {num_losses} losses)")
        logger.info(f"Average Winning Trade: ${avg_win:.2f}")
        logger.info(f"Average Losing Trade: ${avg_loss:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Average PnL per Trade: ${avg_trade_pnl:.2f}")

    logging.info("✅ Validation Complete.")


if __name__ == "__main__":
    main()
