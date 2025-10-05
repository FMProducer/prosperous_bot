# configs/all_tickers_long.py
import torch
from config import cfg

# Set a unique name for this configuration
cfg.paths.config_name = 'all_tickers_long_only'

# --- PATHS ---
cfg.paths.base_output_dir = 'third_party/rl-trading-binance/output'
cfg.paths.train_data_path = 'third_party/rl-trading-binance/data/train_data.npz'
cfg.paths.backtest_data_path = 'third_party/rl-trading-binance/data/backtest_data_20250301_20250601.npz'
cfg.paths.extra_model_dir = 'third_party/rl-trading-binance/output/fmproducer_1_eval/saved_models/session_1'

# --- DEVICE ---
cfg.device.device = torch.device('cpu') # Force CPU usage

# --- BACKTEST SETTINGS ---
cfg.backtest.continuous_data = True  # Use continuous data loader
# cfg.backtest.ticker_name = '1000RATSUSDT'
# Risk & execution (aligned with config.py defaults and dataset spec)
cfg.backtest.volatility_threshold = 0.05         # was 0.10; align with dataset's ≥5% impulse
cfg.backtest.long_action_threshold = 0.012695    # explicit default (config.py)
cfg.backtest.short_action_threshold = 0.013000   # re-enable shorts (was 100.0)
cfg.backtest.close_action_threshold = 0.001141   # explicit default (config.py)
cfg.backtest.use_risk_management = True
cfg.backtest.stop_loss = 0.03
cfg.backtest.take_profit = 0.06
cfg.backtest.trailing_stop = 0.01
cfg.backtest.position_fraction = 0.30
cfg.backtest.max_parallel_sessions = 64          # was 300; reduce commission churn
# cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.selection_strategy = "ensemble_q_filter"