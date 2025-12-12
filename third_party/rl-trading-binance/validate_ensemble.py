import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from trading_environment import TradingEnvironment
from model import DuelingQNetwork
from agent import D3QN_PER_Agent  # Assuming standard agent structure
from utils import load_and_normalize_data
from config import MasterConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- UTILS TO CREATE AGENT STRUCTURE ---
def create_agent(action_dim=3, device='cuda'):
    """
    Creates a dummy agent shell to load weights into.
    Config parameters are hardcoded to match typical training settings.
    """
    # Create a minimal config object required for agent init
    class DummyConfig:
        def __init__(self):
            self.device = device
            # Model params
            self.input_shape = (10, 90) # Standard (Channels, SeqLen)
            self.action_dim = action_dim
            self.dropout = 0.0
            self.hidden_dim = 512 # Default
            # Agent params (not used for inference but needed for init)
            self.lr = 1e-4
            self.gamma = 0.99
            self.tau = 1e-3
            self.batch_size = 64
            self.buffer_size = 1000
            self.prioritized_replay = True
            self.alpha = 0.6
            self.beta_start = 0.4
            self.beta_frames = 1000
            self.use_amp = True
            self.grad_clip = 1.0
            self.mc_dropout = False
            self.mc_samples = 1

    cfg = DummyConfig()
    
    # Initialize networks manually to ensure correct architecture
    policy_net = DuelingQNetwork(
        input_shape=cfg.input_shape, 
        action_dim=cfg.action_dim, 
        dropout=cfg.dropout,
        additional_feats=10 # Hardcoded for compatibility with provided weights
    ).to(device)
    
    target_net = DuelingQNetwork(
        input_shape=cfg.input_shape, 
        action_dim=cfg.action_dim, 
        dropout=cfg.dropout,
        additional_feats=10
    ).to(device)
    
    # We create a hollow agent class to hold the nets
    class SimpleAgent:
        def __init__(self, policy, target, dev):
            self.policy_net = policy
            self.target_net = target
            self.device = dev
            
        def load_model(self, path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load state dict
            if 'policy_net_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            else:
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(checkpoint)
                
            logger.info(f"Model loaded from {path}")

    return SimpleAgent(policy_net, target_net, device)


class EnsembleAgent:
    def __init__(self, long_path, short_path, device, 
                 use_confidence=False, threshold=0.01, 
                 enable_long=True, enable_short=True):
        
        self.device = device
        self.use_confidence = use_confidence
        self.threshold = threshold
        self.enable_long = enable_long
        self.enable_short = enable_short
        
        logger.info(f"🎭 Initializing Ensemble Agent (Hedge Mode + Cross-Close)...")
        logger.info(f"   Conf Mode: {self.use_confidence} | Threshold: {self.threshold}")
        logger.info(f"   Long Enabled: {self.enable_long} | Short Enabled: {self.enable_short}")

        # Load Long Specialist
        if self.enable_long:
            logger.info(f"   Loading LONG: {long_path}")
            self.agent_long = create_agent(action_dim=3, device=device)
            self.agent_long.load_model(long_path)
            self.agent_long.policy_net.eval()
            
        # Load Short Specialist
        if self.enable_short:
            logger.info(f"   Loading SHORT: {short_path}")
            self.agent_short = create_agent(action_dim=3, device=device)
            self.agent_short.load_model(short_path)
            self.agent_short.policy_net.eval()

    def _prepare_state(self, state, mode="LONG"):
        """ Smart Feature Mapping: Adapts 12 Env Feats to 10 Model Feats """
        if not isinstance(state, np.ndarray): state = np.array(state)
        
        DATA_SIZE = 900 # 10 channels * 90 seq
        
        if state.ndim == 1 and state.shape[0] > DATA_SIZE:
            data_part = state[:DATA_SIZE]
            features_part = state[DATA_SIZE:]
            
            # Indices: 0-3 (Base), 4-7 (Step1), 8-11 (Step2)
            # Order: H, L, S, C
            
            if mode == "LONG":
                # Need: [Base] + [H, L, C] (Skip S at 6, 10)
                mask = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]
            else: # SHORT
                # Need: [Base] + [H, S, C] (Skip L at 5, 9)
                mask = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
                
            if len(features_part) >= 12:
                feats = features_part[mask]
                return np.concatenate([data_part, feats])
            else:
                return np.concatenate([data_part, features_part[:10]])
        return state

    def get_long_vote(self, state):
        """ Returns 1 (Open Long) or 0 (Hold). Ignores Close output. """
        if not self.enable_long: return 0
        
        state_mapped = self._prepare_state(state, "LONG")
        with torch.no_grad():
            t_state = torch.from_numpy(state_mapped).float().unsqueeze(0).to(self.agent_long.device)
            q_values = self.agent_long.policy_net(t_state).squeeze(0)
            
        # q_values: [Hold, Open, Close]
        
        if self.use_confidence:
            # Logic with Threshold
            probs = torch.softmax(q_values, dim=0)
            conf_open = (probs[1] - 0.33).item()
            if conf_open > self.threshold and q_values[1] > q_values[0]:
                return 1
        else:
            # Simple Voting (Argmax between Open and Hold)
            # We strictly ignore Index 2 (Close)
            if q_values[1] > q_values[0]:
                return 1
                
        return 0

    def get_short_vote(self, state):
        """ Returns 2 (Open Short) or 0 (Hold). Ignores Close output. """
        if not self.enable_short: return 0
        
        state_mapped = self._prepare_state(state, "SHORT")
        with torch.no_grad():
            t_state = torch.from_numpy(state_mapped).float().unsqueeze(0).to(self.agent_short.device)
            q_values = self.agent_short.policy_net(t_state).squeeze(0)

        # q_values: [Hold, Open (Short), Close]
        
        if self.use_confidence:
            probs = torch.softmax(q_values, dim=0)
            conf_open = (probs[1] - 0.33).item()
            if conf_open > self.threshold and q_values[1] > q_values[0]:
                return 2
        else:
            # Simple Voting
            if q_values[1] > q_values[0]:
                return 2
                
        return 0


def run_validation(config_path):
    # Load Config
    cfg = MasterConfig(config_path)
    device = cfg.device
    
    # Load Data
    sequences, all_stats, keys = load_and_normalize_data(
        data_path=cfg.backtest.val_data_path,
        stats_path=cfg.backtest.norm_stats_path,
        config=cfg,
        is_training=False
    )
    
    # Initialize Environment Params
    # IMPORTANT: Ensure max_steps=60 is set in config or hardcoded here if logic dictates
    env_params = {
        "sequences": sequences,  # Will be set per episode
        "stats": all_stats,      # Will be set per episode
        "initial_balance": cfg.market.initial_balance,
        "commission_rate": cfg.market.commission_rate,
        "slippage": cfg.market.slippage,
        "max_steps": 60,         # HARDCODED: 60 min session as requested
        "window_size": 90,
        "render_mode": None,
        "device": device,
        "reward_config": cfg.reward,
        # Force single direction per env
        "allowed_directions": ['LONG', 'SHORT'], # Placeholder, will be overridden
        "filter_direction": None,
        "normalize": False # Data is already normalized
    }

    # Initialize Ensemble
    # Handle optional config field safely
    use_conf = getattr(cfg.ensemble, 'use_confidence', False)
    thresh = getattr(cfg.ensemble, 'threshold', 0.0)
    
    agent = EnsembleAgent(
        long_path=cfg.ensemble.long_model_path,
        short_path=cfg.ensemble.short_model_path,
        device=device,
        use_confidence=use_conf,
        threshold=thresh,
        enable_long=cfg.ensemble.enable_long,
        enable_short=cfg.ensemble.enable_short
    )
    
    # Create Dual Environments
    env_params_long = env_params.copy()
    env_params_long["allowed_directions"] = ['LONG']
    env_long = TradingEnvironment(**env_params_long)
    
    env_params_short = env_params.copy()
    env_params_short["allowed_directions"] = ['SHORT']
    env_short = TradingEnvironment(**env_params_short)
    
    # Patch total_commission (hotfix)
    env_long.total_commission = 0.0
    env_short.total_commission = 0.0
    
    all_trades = []
    
    logger.info(f"🚀 Starting Validation (Flip-Only, MaxHold=60)...")
    
    for i in tqdm(range(len(sequences)), desc="Simulating"):
        # Setup Episode
        ticker = keys[i].split('_')[0]
        seq = sequences[i]
        stats = all_stats[ticker]
        
        env_long.set_current_sequence(seq, stats)
        env_short.set_current_sequence(seq, stats)
        
        obs_l, _ = env_long.reset()
        obs_s, _ = env_short.reset()
        
        # Reset commission tracking
        if not hasattr(env_long, 'total_commission'): env_long.total_commission = 0.0
        if not hasattr(env_short, 'total_commission'): env_short.total_commission = 0.0
        
        done_l = False
        done_s = False
        
        while not (done_l and done_s):
            # 1. Get Votes (0, 1, or 2) - No Close actions (3)
            # ------------------------------------------------
            vote_l = 0
            if not done_l:
                vote_l = agent.get_long_vote(obs_l) # Returns 1 or 0
                
            vote_s = 0
            if not done_s:
                vote_s = agent.get_short_vote(obs_s) # Returns 2 or 0
                
            # 2. Conflict Resolution & Cross-Closing
            # ---------------------------------------
            final_act_l = vote_l
            final_act_s = vote_s
            
            # Conflict: Both want to enter at same time
            if vote_l == 1 and vote_s == 2:
                # UNCERTAINTY -> FLAT ALL
                # Если у нас есть позиции - закрываем их принудительно.
                # Новые не открываем.
                
                final_act_l = 3 if env_long.position > 0 else 0
                final_act_s = 3 if env_short.position < 0 else 0
            
            # Cross-Close Logic
            elif vote_l == 1:
                # Long entry -> Close Short if exists
                if env_short.position < 0:
                    final_act_s = 3 # Force Close Short
            
            elif vote_s == 2:
                # Short entry -> Close Long if exists
                if env_long.position > 0:
                    final_act_l = 3 # Force Close Long
            
            # 3. Execution
            # ------------
            
            # -- Long Env --
            if not done_l:
                signal_dt = env_long.df.index[env_long.current_step]
                next_obs, _, term, trunc, info = env_long.backtest_step(
                    action=final_act_l, signal_dt=signal_dt, ticker=ticker
                )
                obs_l = next_obs
                done_l = term or trunc
                
                # Capture Trades
                if info.get('position_closed'):
                    t_data = info.copy()
                    t_data['symbol'] = f"{ticker}_L"
                    all_trades.append(t_data)
            
            # -- Short Env --
            if not done_s:
                signal_dt = env_short.df.index[env_short.current_step]
                next_obs, _, term, trunc, info = env_short.backtest_step(
                    action=final_act_s, signal_dt=signal_dt, ticker=ticker
                )
                obs_s = next_obs
                done_s = term or trunc
                
                # Capture Trades
                if info.get('position_closed'):
                    t_data = info.copy()
                    t_data['symbol'] = f"{ticker}_S"
                    all_trades.append(t_data)

    # --- FINAL METRICS ---
    print("\n" + "="*44)
    print("📊 FINAL VALIDATION RESULTS (HEDGE + FLIP)")
    print("="*44)
    
    if not all_trades:
        print("Trades: 0 (No signals triggered)")
        return

    df_trades = pd.DataFrame(all_trades)
    
    # Safe float conversion
    for col in ['trade_realized_pnl', 'trade_commission', 'holding_duration_bars']:
        if col in df_trades.columns:
            df_trades[col] = df_trades[col].astype(float)
            
    # Calculate Net PnL
    df_trades['net_pnl'] = df_trades['trade_realized_pnl'] - df_trades.get('trade_commission', 0.0)
    
    total_trades = len(df_trades)
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_net_pnl = df_trades['net_pnl'].sum()
    total_comm = df_trades['trade_commission'].sum()
    
    gross_win = wins['net_pnl'].sum()
    gross_loss = abs(losses['net_pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {pf:.4f}")
    print(f"Net PnL: {total_net_pnl:.2f} USDT")
    print(f"Total Commission: {total_comm:.2f} USDT")
    print(f"Avg Hold: {df_trades['holding_duration_bars'].mean():.1f} bars")
    print("="*44)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--ensemble", action="store_true", help="Run ensemble validation")
    args = parser.parse_args()
    
    if args.ensemble:
        run_validation(args.config)
    else:
        print("Please use --ensemble flag to run this validator.")