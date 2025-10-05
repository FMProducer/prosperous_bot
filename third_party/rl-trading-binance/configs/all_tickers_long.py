# configs/all_tickers_long.py
from config import cfg

# Set a unique name for this configuration
cfg.paths.config_name = 'all_tickers_long_only'

# --- PATHS ---
cfg.paths.base_output_dir = 'third_party/rl-trading-binance/output'
cfg.paths.train_data_path = 'third_party/rl-trading-binance/data/train_data.npz'
cfg.paths.backtest_data_path = 'third_party/rl-trading-binance/data/backtest_data_20250301_20250601.npz'
cfg.paths.extra_model_dir = 'third_party/rl-trading-binance/output/fmproducer_1_eval/saved_models/session_1'

# --- DEVICE ---
cfg.device.device = 'cpu' # Force CPU usage

# --- BACKTEST SETTINGS ---
cfg.backtest.continuous_data = True # Use the new continuous data loader
# cfg.backtest.ticker_name = '1000RATSUSDT' # Commented out to run on all tickers from tickers.txt
cfg.backtest.volatility_threshold = 0.10 # 10% volatility filter
cfg.backtest.short_action_threshold = 100.0 # Disable shorting to run long-only strategy