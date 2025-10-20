import torch
import pandas as pd
import numpy as np
import datetime as dt

from model import DuelingQNetwork
from paper_trader import Cfg
from utils import calculate_normalization_stats, apply_normalization
from config import MasterConfig

class DuelingQPolicy:
    def __init__(self, model, stats, master_cfg):
        self.model = model
        self.stats = stats
        self.master_cfg = master_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, symbol: str, ctx_df: pd.DataFrame) -> str:
        # 1. Prepare the observation from ctx_df
        # This needs to replicate the logic from TradingEnvironment._get_observation
        
        # The ctx_df is the window. Its length should be agent_history_len
        window = ctx_df[self.master_cfg.data.expected_channels].to_numpy(dtype=np.float32)

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
        
        action = int(np.argmax(qvals))

        # 3. Map action to string
        if action == 1:
            return "BUY"
        elif action == 2:
            return "SELL"
        else:
            return "HOLD"

def load_policy(ckpt_path: str, cfg: Cfg):
    # This is a simplified way to get the master config.
    # It assumes that the paper_trader config has enough information.
    master_cfg = MasterConfig()
    # We need to populate master_cfg with values from cfg if they exist.
    # For now, we will use the defaults from config.py and alpha.py,
    # as they are loaded in train.py.
    
    # A better way would be to load the config file that was used for training.
    # Assuming the config is compatible.
    
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

    # This is the hardest part: getting the normalization stats.
    # For now, I will create dummy stats.
    # In a real scenario, these should be loaded from a file saved during training.
    stats = {
        "means": {ch: 0.0 for ch in master_cfg.data.data_channels},
        "stds": {ch: 1.0 for ch in master_cfg.data.data_channels},
    }

    return DuelingQPolicy(model, stats, master_cfg)