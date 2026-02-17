import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
import unittest.mock as mock

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("❌ Matplotlib is missing. Please run: pip install matplotlib")
    sys.exit(1)

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "third_party" / "rl-trading-binance"
sys.path.append(str(PROJECT_ROOT))

# Import Strategy
STRATEGY_PATH = PROJECT_ROOT / "user_data" / "strategies" / "CustomD3QNStrategy4z.py"
spec = importlib.util.spec_from_file_location("CustomD3QNStrategy4z", STRATEGY_PATH)
strat_mod = importlib.util.module_from_spec(spec)
sys.modules["CustomD3QNStrategy4z"] = strat_mod
spec.loader.exec_module(strat_mod)
CustomD3QNStrategy4z = strat_mod.CustomD3QNStrategy4z

class QValueVisualizer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            self.config_path = PROJECT_ROOT / "user_data" / "config_rl4z.json"

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.config["rl_calibration_mode"] = True
        self.config["runmode"] = "backtest"
        self.config["deep_inference"] = True

        self._setup_strategy_environment()
        self.strategy = CustomD3QNStrategy4z(self.config)

    def _setup_strategy_environment(self):
        """Mock environment to allow strategy loading without real weights."""
        example_path = PROJECT_ROOT / "output/alpha_seed_404_ohlcv_z_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260125_time_033653/best.pth"
        if not example_path.exists():
            def get_mock_agent_cfg(path):
                m_cfg = mock.Mock()
                m_cfg.market.num_actions = 3
                m_cfg.market.mirror_mode = False
                m_cfg.seq.state_shape = (5, 90, 1)
                # Minimal params to pass checks
                m_cfg.model.cnn_maps = [16]
                m_cfg.model.dense_val = [64]
                return m_cfg

            CustomD3QNStrategy4z._find_config_file = lambda s, d: Path("mock.py")
            CustomD3QNStrategy4z._load_py_config = lambda s, p: get_mock_agent_cfg(p)
            CustomD3QNStrategy4z._load_weights = lambda s, a, p, n: None

    def load_data(self, periods=10000):
        real_data_path = REPO_ROOT / "freqtrade/tests/testdata/UNITTEST_BTC-1m.json"
        if real_data_path.exists():
            print(f"Loading real data from {real_data_path}")
            with open(real_data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            if len(df) > periods:
                df = df.tail(periods).reset_index(drop=True)
            return self.strategy.populate_indicators(df, {"pair": "BTC/USDT"})
        return pd.DataFrame()

    def get_raw_advantages(self, df):
        print("Running inference...")
        self.strategy.populate_entry_trend(df.copy(), {"pair": "BTC/USDT"})
        
        last_date = df.iloc[-1]['date']
        q_cache_key = ("BTC/USDT", str(last_date))
        q_values = self.strategy.q_value_cache.get(q_cache_key)

        if not q_values:
            print("❌ No Q-values found in cache.")
            return [], []

        long_advs = []
        short_advs = []
        norm_stats = self.strategy.q_normalization

        for name in ["long_1", "long_2", "short_1", "short_2"]:
            if name in q_values:
                q = q_values[name]
                # Determine action index (Long=1, Short=2 usually, but depends on mirror)
                if name.startswith("long"):
                    action_idx = 1
                else:
                    action_idx = 1 if getattr(self.strategy, f"{name}_is_mirror", False) else 2

                # Raw Advantage: Q(Action) - Q(Hold)
                adv = q[:, action_idx] - q[:, 0]

                # Normalize
                stats = norm_stats.get(name, {})
                q_min = stats.get('q_min', 0.0)
                q_max = stats.get('q_max', 1.0)
                
                if q_max > q_min:
                    norm = (adv - q_min) / (q_max - q_min)
                    norm = np.clip(norm, 0.0, 1.0)
                    
                    if name.startswith("long"):
                        long_advs.extend(norm)
                    else:
                        short_advs.extend(norm)

        return np.array(long_advs), np.array(short_advs)

    def plot(self):
        df = self.load_data()
        if df.empty:
            return

        longs, shorts = self.get_raw_advantages(df)
        
        plt.figure(figsize=(10, 6))
        plt.hist(longs, bins=50, alpha=0.6, label=f'Long Advantages (n={len(longs)})', color='green')
        plt.hist(shorts, bins=50, alpha=0.6, label=f'Short Advantages (n={len(shorts)})', color='red')
        
        # Plot current thresholds
        th_l = self.config.get("rl_long_threshold", 0.3)
        th_s = self.config.get("rl_short_threshold", 0.3)
        plt.axvline(th_l, color='darkgreen', linestyle='dashed', linewidth=2, label=f'Long Thresh ({th_l})')
        plt.axvline(th_s, color='darkred', linestyle='dashed', linewidth=2, label=f'Short Thresh ({th_s})')

        plt.title("Distribution of Normalized Q-Advantages")
        plt.xlabel("Advantage Strength (0.0 = Weak, 1.0 = Strong)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = "q_distribution.png"
        plt.savefig(output_file)
        print(f"✅ Plot saved to {output_file}")

if __name__ == "__main__":
    QValueVisualizer("user_data/config_rl4z.json").plot()