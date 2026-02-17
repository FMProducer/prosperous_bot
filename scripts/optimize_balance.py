import json
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import torch
import importlib.util
import unittest.mock as mock

# Setup paths to import the strategy and its dependencies
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "third_party" / "rl-trading-binance"
sys.path.append(str(PROJECT_ROOT))

# Import CustomD3QNStrategy4z
STRATEGY_PATH = PROJECT_ROOT / "user_data" / "strategies" / "CustomD3QNStrategy4z.py"
spec = importlib.util.spec_from_file_location("CustomD3QNStrategy4z", STRATEGY_PATH)
strat_mod = importlib.util.module_from_spec(spec)
sys.modules["CustomD3QNStrategy4z"] = strat_mod
spec.loader.exec_module(strat_mod)
CustomD3QNStrategy4z = strat_mod.CustomD3QNStrategy4z

class BalanceOptimizer:
    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            # Fallback for repository root execution
            self.config_path = PROJECT_ROOT / "user_data" / "config_rl4z.json"

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Enable calibration mode and offline run
        self.config["rl_calibration_mode"] = True
        self.config["runmode"] = "backtest"
        self.config["deep_inference"] = False

        # Mock models to ensure script runs in environments with missing weights
        self._setup_strategy_environment()

        self.strategy = CustomD3QNStrategy4z(self.config)

    def _setup_strategy_environment(self):
        """
        Setup mocks for strategy initialization if real models are missing.
        Uses monkeypatching on the class before instantiation.
        """
        # Check if one of the model files exists as a proxy for environment health
        example_path = PROJECT_ROOT / "output/alpha_seed_404_ohlcv_z_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260125_time_033653/best.pth"
        if not example_path.exists():
            print("ℹ️ Real models not found. Enabling mock mode for strategy initialization.")

            def get_mock_agent_cfg(path):
                m_cfg = mock.Mock()
                m_cfg.market.num_actions = 3
                m_cfg.market.mirror_mode = False
                m_cfg.seq.state_shape = (5, 90, 1)

                params = {
                    'model.cnn_maps': [16], 'model.cnn_kernels': [3], 'model.cnn_strides': [1], 'model.cnn_dilations': [1],
                    'model.dense_val': [64], 'model.dense_adv': [64], 'model.additional_feats': 4, 'model.dropout_p': 0.0,
                    'rl.gamma': 0.99, 'rl.lr': 1e-4, 'rl.batch_size': 32, 'rl.target_update_freq': 100,
                    'rl.train_start': 1000, 'rl.max_gradient_norm': 1.0, 'per.buffer_size': 10000,
                    'per.per_alpha': 0.6, 'per.per_beta_start': 0.4, 'per.per_beta_frames': 10000,
                    'eps.eps_start': 1.0, 'eps.eps_end': 0.1, 'eps.eps_decay_frames': 10000
                }

                for attr, val in params.items():
                    target = m_cfg
                    parts = attr.split('.')
                    for part in parts[:-1]:
                        if not hasattr(target, part):
                            setattr(target, part, mock.Mock())
                        target = getattr(target, part)
                    setattr(target, parts[-1], val)
                return m_cfg

            CustomD3QNStrategy4z._find_config_file = lambda s, d: Path("mock.py")
            CustomD3QNStrategy4z._load_py_config = lambda s, p: get_mock_agent_cfg(p)
            CustomD3QNStrategy4z._load_weights = lambda s, a, p, n: None

    def load_data_sample(self, pair: str, periods: int = 10000) -> pd.DataFrame:
        """
        Load real freqtrade test data or generate synthetic candles.
        """
        real_data_path = REPO_ROOT / "freqtrade/tests/testdata/UNITTEST_BTC-1m.json"
        if real_data_path.exists():
            print(f"Loading real data from {real_data_path}")
            with open(real_data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            if len(df) > periods:
                df = df.tail(periods).reset_index(drop=True)
            print(f"Loaded {len(df)} candles.")
        else:
            print(f"Generating {periods} synthetic candles for {pair}...")
            dates = pd.date_range(end=pd.Timestamp.now('UTC'), periods=periods, freq="1min")
            df = pd.DataFrame({"date": dates})
            np.random.seed(42)
            rets = np.random.normal(0.0, 0.001, size=periods)
            price = 100.0 * np.exp(np.cumsum(rets))
            df["open"] = price
            df["close"] = price * (1.0 + np.random.normal(0.0, 0.0002, size=periods))
            df["high"] = df[["open", "close"]].max(axis=1) * (1.0 + np.abs(np.random.normal(0.0, 0.0005, size=periods)))
            df["low"] = df[["open", "close"]].min(axis=1) * (1.0 - np.abs(np.random.normal(0.0, 0.0005, size=periods)))
            df["volume"] = np.random.rand(periods) * 1000.0

        meta = {"pair": pair}
        return self.strategy.populate_indicators(df, meta)

    def get_signals_matrix(self, df: pd.DataFrame, pair: str) -> Dict[str, np.ndarray]:
        """
        Extract normalized advantages for long and short sides from the strategy.
        """
        print("Running strategy inference...")
        meta = {"pair": pair}
        self.strategy.populate_entry_trend(df.copy(), meta)

        last_date = df.iloc[-1]['date']
        q_cache_key = (pair, str(last_date))
        q_values = self.strategy.q_value_cache.get(q_cache_key)

        if q_values is None:
            raise ValueError("Inference failed: Q-values not found in strategy cache.")

        norm_stats = self.strategy.q_normalization
        norm_advs = {}

        for name in ["long_1", "long_2", "short_1", "short_2"]:
            if name in q_values:
                q = q_values[name]
                if name.startswith("long"):
                    action_idx = 1
                else:
                    action_idx = 1 if getattr(self.strategy, f"{name}_is_mirror", False) else 2

                # Vectorized advantage calculation: Q(Action) - Q(Hold)
                adv = q[:, action_idx] - q[:, 0]

                # Vectorized Z-normalization based on strategy's stats
                stats = norm_stats.get(name, {})
                q_min = stats.get('q_min', 0.0)
                q_max = stats.get('q_max', q_min)

                if q_max > q_min:
                    norm = (adv - q_min) / (q_max - q_min)
                else:
                    norm = np.zeros_like(adv)

                norm_advs[name] = np.clip(norm, 0.0, 1.0)
            else:
                batch_size = next(iter(q_values.values())).shape[0]
                norm_advs[name] = np.zeros(batch_size)

        thresh_l = self.config.get("rl_long_threshold", 1)
        thresh_s = self.config.get("rl_short_threshold", 1)

        # Combine model advantages per side according to voting threshold (thresh=2 -> min, thresh=1 -> max)
        long_side = np.minimum(norm_advs["long_1"], norm_advs["long_2"]) if thresh_l >= 2 else np.maximum(norm_advs["long_1"], norm_advs["long_2"])
        short_side = np.minimum(norm_advs["short_1"], norm_advs["short_2"]) if thresh_s >= 2 else np.maximum(norm_advs["short_1"], norm_advs["short_2"])

        pad = len(df) - len(long_side)
        return {
            "long": np.pad(long_side, (pad, 0)).astype(np.float32),
            "short": np.pad(short_side, (pad, 0)).astype(np.float32),
        }

    def evaluate_metric(self, df: pd.DataFrame, signals: Dict[str, np.ndarray], eps_l: float, eps_s: float) -> Tuple[float, float, int]:
        """
        Calculate V-Diff, PnL-Diff and Total Signals for given thresholds.
        """
        mask_l = signals["long"] > eps_l
        mask_s = signals["short"] > eps_s

        v_l, v_s = int(mask_l.sum()), int(mask_s.sum())
        total = v_l + v_s
        if total == 0:
            return 1.0, 1.0, 0

        v_diff = abs(v_l - v_s) / total

        # Proxy PnL using future returns over 120m horizon
        horizon = 60  # Sync with strategy timeout_60m
        close = df["close"].to_numpy()
        future_ret = np.zeros_like(close)
        future_ret[:-horizon] = (close[horizon:] / close[:-horizon]) - 1.0

        pnl_l = float(future_ret[mask_l].sum())
        pnl_s = float(-future_ret[mask_s].sum())

        denom = abs(pnl_l) + abs(pnl_s)
        # PnL-Diff: Balance of profit contribution (we want L and S to contribute equally)
        p_diff = abs(pnl_l - pnl_s) / denom if denom > 1e-6 else 1.0

        return v_diff, p_diff, total

    def optimize(self) -> None:
        """
        Execute grid search to find best balanced epsilon thresholds.
        """
        if not self.strategy.q_normalization:
            print("⚠️ WARNING: 'q_normalization' is empty in config! Optimization requires pre-calculated stats.")
            print("   Run the strategy in dry-run/live first or populate config_rl4z.json manually.")

        pair = "BTC/USDT:USDT"
        df = self.load_data_sample(pair)
        signals = self.get_signals_matrix(df, pair)

        print(f"\n{'EpsL':<8} {'EpsS':<8} {'V-Diff':<10} {'PnL-Diff':<10} {'Total':<8} Status")
        print("-" * 65)

        candidates = []
        for eps_l in np.arange(0.10, 0.99, 0.02):
            for eps_s in np.arange(0.10, 0.99, 0.02):
                v_diff, p_diff, total = self.evaluate_metric(df, signals, eps_l, eps_s)
                ok = (v_diff <= 0.15) and (p_diff <= 0.20) and (total >= 50)
                if ok:
                    candidates.append((eps_l, eps_s, v_diff, p_diff, total))

                if ok or (int(eps_l*100) % 10 == 0 and int(eps_s*100) % 10 == 0):
                    print(f"{eps_l:<8.2f} {eps_s:<8.2f} {v_diff:<10.2%} {p_diff:<10.2%} {total:<8d} {'✅' if ok else '..'}")

        if not candidates:
            print("\nNo balanced configuration found. Consider widening search ranges or reviewing signals.")
            return

        # OPTION 1: Prioritize V-Diff (Volume Balance) above everything else
        # candidates.sort(key=lambda x: x[2])
        
        # OPTION 2: Prioritize Total Signals (find the most active balanced config)
        # candidates.sort(key=lambda x: x[4], reverse=True)

        # CURRENT: Minimize sum of V-Diff and PnL-Diff (Balanced approach)
        candidates.sort(key=lambda x: x[2] + x[3])
        
        best_l, best_s, v_diff, p_diff, total = candidates[0]

        print("\nBest balanced configuration:")
        print(f"  epsilon_threshold_long  = {best_l:.3f}")
        print(f"  epsilon_threshold_short = {best_s:.3f}")
        print(f"  V-Diff  = {v_diff:.2%}")
        print(f"  PnL-Diff = {p_diff:.2%}")
        print(f"  Total signals = {total}")

        self.update_config(best_l, best_s)

    def update_config(self, best_l: float, best_s: float) -> None:
        """
        Update the configuration file with the optimized thresholds.
        """
        print(f"\nSaving optimized thresholds to {self.config_path}...")
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            data["rl_long_threshold"] = round(best_l, 3)
            data["rl_short_threshold"] = round(best_s, 3)
            
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print("✅ Configuration updated.")
        except Exception as e:
            print(f"❌ Failed to update config: {e}")

def main():
    config_path = "user_data/config_rl4z.json"
    BalanceOptimizer(config_path).optimize()

if __name__ == "__main__":
    main()
