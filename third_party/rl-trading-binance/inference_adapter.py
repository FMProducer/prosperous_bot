import torch
import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Any

from model import DuelingQNetwork
from utils import calculate_normalization_stats, apply_normalization
from config import MasterConfig

class DuelingQPolicy:
    def __init__(self, model, stats, master_cfg):
        self.model = model
        self.stats = stats
        self.master_cfg = master_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, symbol: str, ctx_df: pd.DataFrame) -> str:
        # 1. Prepare the observation from ctx_df
        # This needs to replicate the logic from TradingEnvironment._get_observation
        
        # The ctx_df is the window. Its length should be agent_history_len

        # Create a copy to avoid SettingWithCopyWarning.
        df = ctx_df.copy()

        # Ensure all expected channels are present in the DataFrame.
        # The data provider might omit optional channels like 'volume_weighted_average' or 'num_trades'
        # if they are not available in the database for a given window.
        # We add them here with a default value of 0.0 to ensure the model's input shape is always correct.
        for channel in self.master_cfg.data.expected_channels:
            if channel not in df.columns:
                df[channel] = 0.0
        
        window = df[self.master_cfg.data.expected_channels].to_numpy(dtype=np.float32)

        # Normalize the window
        normalized_window = apply_normalization(
            window,
            self.stats,
            self.master_cfg.data.data_channels,
            self.master_cfg.data.price_channels,
            self.master_cfg.data.volume_channels,
            self.master_cfg.data.other_channels,
            self.master_cfg.seq.agent_history_len,
            self.master_cfg.seq.input_history_len,
        )

        # Extras: position, unrealized_pnl, time_elapsed, time_remaining
        extras = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Action history
        hist_onehot = np.zeros(self.master_cfg.seq.action_history_len * self.master_cfg.market.num_actions, dtype=np.float32)

        # Combine to form the state
        state = np.concatenate([normalized_window.flatten(), extras, hist_onehot])
        
        # 2. Predict
        with torch.no_grad():
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            qvals = self.model(tensor).cpu().numpy().squeeze(0)
        
        # --- Advantage-based filtering logic from backtest_engine.py ---
        # Advantage = Q(s,a) - V(s). V(s) is approximated by Q(s, a=0/hold).
        adv = qvals - qvals[0]
        action = int(np.argmax(adv))
        confidence = adv[action]

        # Check against thresholds from config
        if action == 1 and confidence < self.master_cfg.backtest.long_action_threshold:
            action = 0  # Reject, set to HOLD
        elif action == 2 and confidence < self.master_cfg.backtest.short_action_threshold:
            action = 0  # Reject, set to HOLD
        
        # 3. Map action to string
        if action == 1:
            return "BUY"
        elif action == 2:
            return "SELL"
        else: # action == 0
            return "HOLD"

def load_policy(ckpt_path: str, master_cfg: MasterConfig, stats: Dict[str, Any]):
    # Используем master_cfg и stats, которые передаёт paper_trader.
    
    # Instantiate the model
    model = DuelingQNetwork(
        input_shape=(master_cfg.seq.num_features, master_cfg.seq.input_history_len, 1),
        action_dim=master_cfg.market.num_actions,
        cnn_maps=master_cfg.model.cnn_maps,
        cnn_kernels=master_cfg.model.cnn_kernels,
        cnn_strides=master_cfg.model.cnn_strides,
        dense_val=master_cfg.model.dense_val,
        dense_adv=master_cfg.model.dense_adv,
        additional_feats=master_cfg.model.additional_feats,
        dropout_p=master_cfg.model.dropout_p,
    )

    # Load the model weights
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if 'policy_state' in checkpoint:
        model.load_state_dict(checkpoint['policy_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Используем переданные статистики, а не заглушку.
    if not stats or "means" not in stats or "stds" not in stats:
        raise ValueError("Normalization stats are missing or invalid.")

    return DuelingQPolicy(model, stats, master_cfg)