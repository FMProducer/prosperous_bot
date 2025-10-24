```diff
--- a/third_party/rl-trading-binance/backtest_engine.py
+++ b/third_party/rl-trading-binance/paper_trader.py
@@ -6,9 +6,14 @@
 from collections import defaultdict
 from typing import Any, Dict, List, Tuple
 
+import matplotlib.pyplot as plt
 import numpy as np
+import pandas as pd
+import psycopg2
+from psycopg2.extras import RealDictCursor
 
 from config import MasterConfig
 from config import cfg as default_cfg
@@ -16,8 +21,9 @@
 from trading_environment import TradingEnvironment
 from utils import (
     calculate_normalization_stats,
-    create_signal_groups,
-    load_config,
-    load_npz_dataset,
+    create_signal_groups,
+    find_spike_windows,
+    load_config,
     select_and_arrange_channels,
     set_random_seed,
 )
@@ -145,19 +151,19 @@
         std_pnl_all_neg = pnl_all[pnl_all < 0].std() if np.any(pnl_all < 0) else 0.0
 
         return {
-            "total_commission": f"{{(-self.total_commission / balances[0]) * 100:.2f}}%" if balances[0] != 0 else "0.00%",
-            "avg_commission": f"{{-self.total_commission / self.total_trades:.2f}}" if self.total_trades > 0 else "0.00",
+            "total_commission": f"{(-self.total_commission / balances[0]) * 100:.2f}%" if balances[0] != 0 else "0.00%",
+            "avg_commission": f"{-self.total_commission / self.total_trades:.2f}" if self.total_trades > 0 else "0.00",
             "max_loss": f"{pnl_all.min():.2f}" if len(pnl_all) > 0 else "0.00",
             "max_profit": f"{pnl_all.max():.2f}" if len(pnl_all) > 0 else "0.00",
             "total_trade_days": trade_days,
             "profit_days": (
-                f"{int((pnl_by_day > 0).sum())} ({{((pnl_by_day > 0).sum() / trade_days) * 100:.2f}}%)"
+                f"{int((pnl_by_day > 0).sum())} ({(pnl_by_day > 0).sum() / trade_days * 100:.2f}%)"
                 if trade_days > 0
                 else "0 (0.00%)"
             ),
-            "final_balance_change": f"{{(total_change - 1) * 100:.2f}}%",
+            "final_balance_change": f"{(total_change - 1) * 100:.2f}%",
             "exp_day_change": (
-                f"{{(np.power(total_change, 1 / trade_days) - 1) * 100:.2f}}%" if trade_days > 0 else "0.00%"
+                f"{(np.power(total_change, 1 / trade_days) - 1) * 100:.2f}%" if trade_days > 0 else "0.00%"
             ),
             "max_drawdown": f"{min(self.drawdowns) * 100:.2f}%" if self.drawdowns else "0.00%",
             "sharpe": (
-                f"{{(pnl_by_day.mean() / (pnl_by_day.std() + 1e-9)) * np.sqrt(len(pnl_by_day)):.2f}}"
+                f"{(pnl_by_day.mean() / (pnl_by_day.std() + 1e-9)) * np.sqrt(len(pnl_by_day)):.2f}"
                 if len(pnl_by_day) > 0
                 else "0.00"
             ),
@@ -165,19 +171,19 @@
                 if len(pnl_by_day) > 0
                 else "0.00"
             ),
-            "trades_sharpe": (f"{pnl_all.mean() / (pnl_all.std() + 1e-9):.2f}" if len(pnl_all) > 0 else "0.00"),
-            "trades_sortino": (f"{pnl_all.mean() / (std_pnl_all_neg + 1e-9):.2f}" if len(pnl_all) > 0 else "0.00"),
-            "accuracy": (f"{self.correct_preds / self.total_trades * 100:.1f}%" if self.total_trades > 0 else "0.0%"),
+            "trades_sharpe": (f"{(pnl_all.mean() / (pnl_all.std() + 1e-9)):.2f}" if len(pnl_all) > 0 else "0.00"),
+            "trades_sortino": (
+                f"{(pnl_all.mean() / std_pnl_all_neg):.2f}" if len(pnl_all) > 0 and std_pnl_all_neg > 1e-9
+                else "0.00"
+            ),
+            "accuracy": (f"{self.correct_preds / self.total_trades * 100:.1f}%" if self.total_trades > 0 else "0.0%"),
             "total_trades": self.total_trades,
             "total_longs": self.total_longs,
             "total_shorts": self.total_shorts,
             "longs_correct": (
-                f"{self.correct_longs} (0.0%)"
+                f"{self.correct_longs} (0.0%)"
                 if self.total_longs == 0
-                else f"{self.correct_longs} ({{(self.correct_longs / self.total_longs) * 100:.1f}}%)"
+                else f"{self.correct_longs} ({(self.correct_longs / self.total_longs) * 100:.1f}%)"
             ),
             "shorts_correct": (
-                f"{self.correct_shorts} (0.0%)"
+                f"{self.correct_shorts} (0.0%)"
                 if self.total_shorts == 0
-                else f"{self.correct_shorts} ({{(self.correct_shorts / self.total_shorts) * 100:.1f}}%)"
+                else f"{self.correct_shorts} ({(self.correct_shorts / self.total_shorts) * 100:.1f}%)"
             ),
             "correct_avg_change": (f"{np.mean(changes[changes > 0]) * 100:.2f}%" if np.any(changes > 0) else "0.00%"),
             "correct_std_change": (f"{np.std(changes[changes > 0]) * 100:.2f}%" if np.any(changes > 0) else "0.00%"),
@@ -211,21 +217,69 @@
     pass_adv = long_pass or short_pass or close_pass
     return pass_adv
 
+# +++ Add a new function to load data from the database and prepare signals.
def load_from_db_and_prepare_signals(cfg: MasterConfig, cfg_mod: Any) -> List[Tuple[Tuple[str, dt.datetime], np.ndarray]]:
     dsn = cfg.db.dsn
     start_utc = cfg_mod.data["time_range"]["start_utc"]
     end_utc = cfg_mod.data["time_range"]["end_utc"]
     symbols = cfg_mod.data["symbols"]
+
     start_ts = int(pd.to_datetime(start_utc, utc=True).timestamp() * 1000)
     end_ts = int(pd.to_datetime(end_utc, utc=True).timestamp() * 1000)
+
     conn = psycopg2.connect(dsn)
     cur = conn.cursor(cursor_factory=RealDictCursor)
+
     backtest_raw = []
+
     for symbol in symbols:
         logging.info(f"Loading data for {symbol}...")
         cur.execute(
             f"SELECT ts, open, high, low, close, volume, volume_weighted_average, num_trades "
-            f"FROM v_klines_1m_npz WHERE symbol = %s AND ts >= %s AND ts < %s ORDER BY ts ASC;",
+            f"FROM v_klines_1m_npz WHERE symbol = %s AND ts >= %s AND ts < %s ORDER BY ts ASC;",
             (symbol, start_ts, end_ts)
         )
         rows = cur.fetchall()
         if not rows:
             logging.warning(f"No data for symbol {symbol} in the given time range.")
             continue
+
         df = pd.DataFrame(rows)
         df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
         df = df.set_index('ts')
+
+        logging.info(f"Finding spike windows for {symbol}...")
+        spike_windows = find_spike_windows(
+            df,
+            context_minutes=cfg_mod.data["detector"]["context_minutes"],
+            window_minutes=cfg_mod.data["detector"]["window_minutes"],
+            abs_change_threshold_pct=cfg_mod.data["detector"]["abs_change_pct"],
+            contrast_min=cfg_mod.data["detector"]["contrast_min"],
+            cooldown_minutes=cfg_mod.data["detector"]["cooldown_minutes"],
+            use_lookahead=cfg_mod.data["detector"]["use_lookahead"],
+        )
+
+        logging.info(f"Found {len(spike_windows)} spike windows for {symbol}.")
+
+        for _, _, session_start, _, _ in spike_windows:
+            signal_dt = session_start
+            seq_start = signal_dt - dt.timedelta(minutes=cfg.seq.pre_signal_len)
+            seq_end = signal_dt + dt.timedelta(minutes=cfg.seq.post_signal_len)
+
+            seq_df = df[(df.index >= seq_start) & (df.index < seq_end)]
+
+            if len(seq_df) == cfg.seq.full_seq_len:
+                seq_arr = seq_df[cfg.data.expected_channels].to_numpy(dtype=np.float32)
+                backtest_raw.append(((symbol, signal_dt), seq_arr))
+
     conn.close()
     return backtest_raw
 
-def run_backtest(cfg: MasterConfig, model_path_override: str = None) -> Dict[str, Any]:
+def run_backtest(cfg: MasterConfig, cfg_mod: Any, model_path_override: str = None) -> Dict[str, Any]:
     cfg.backtest_mode = True
     setup_logging(cfg)
     set_random_seed(cfg.random_seed)
 
-    backtest_raw = load_npz_dataset(
-        file_path=cfg.paths.backtest_data_path,
-        name_dataset="Backtest",
-        plot_dir=cfg.paths.plot_dir,
-        debug_max_size=cfg.debug.debug_max_size_data,
-        plot_examples=cfg.data.plot_examples,
-        plot_channel_idx=cfg.data.plot_channel_idx,
-        pre_signal_len=cfg.seq.pre_signal_len,
-    )
+
+    # --- Change data loading from NPZ to DB
+    backtest_raw = load_from_db_and_prepare_signals(cfg, cfg_mod)
 
     grouped_backtest_data = create_signal_groups(backtest_raw)
 
@@ -238,29 +292,10 @@
     if os.path.exists(stats_path):
         logging.info(f"Loading normalization stats from {stats_path}")
         with open(stats_path, 'r') as f:
-            stats = json.load(f)
+            stats = json.load(f)
 
     if stats is None:
-        logging.info("Normalization stats not found, calculating...")
-        train_raw = load_npz_dataset(
-            file_path=cfg.paths.train_data_path,
-            name_dataset="Train",
-            plot_dir=cfg.paths.plot_dir,
-            debug_max_size=cfg.debug.debug_max_size_data,
-            plot_examples=0,
-            plot_channel_idx=None,
-            pre_signal_len=cfg.seq.pre_signal_len,
-        )
-        train_seqs = []
-        for _, arr in train_raw:
-            sel = select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels)
-            if sel is not None:
-                train_seqs.append(sel)
-        stats = calculate_normalization_stats(
-            train_seqs,
-            cfg.data.data_channels,
-            cfg.data.price_channels,
-            cfg.data.volume_channels,
-            cfg.data.other_channels,
-        )
-        with open(stats_path, 'w') as f:
-            json.dump(stats, f, indent=2)
-        logging.info(f"Normalization stats saved to {stats_path}")
+        # --- The logic to calculate stats from train data is removed
+        # --- because we are not loading train data in this script.
+        # --- We assume the stats file is pre-generated.
+        logging.error("Normalization stats not found. Please generate them first.")
+        raise RuntimeError("Normalization stats not found. Please generate them first.")
 
     if model_path_override:
         model_path = model_path_override
@@ -342,8 +397,8 @@
                     pass_adv = get_pass_advantage(action, confidence, cfg)
                     if pass_adv:
                         logging.info(
-                            f": REJECTED {{['LONG', 'SHORT', 'CLOSE'][action-1]}}, "
-                            f"confidence={{confidence:.3f}} < threshold={thresholds[action-1]}"
+                            f": REJECTED {['LONG', 'SHORT', 'CLOSE'][action-1]}, "
+                            f"confidence={confidence:.3f} < threshold={thresholds[action-1]}"
                         )
                         action = 0
                 # MC-Dropout (Monte Carlo Dropout)
@@ -363,9 +418,9 @@
                     pass_uncertainty = uncertainty >= cfg.backtest.ensemble_max_sigma
                     if pass_adv and pass_uncertainty:
                         logging.info(
-                            f": REJECTED {{['LONG', 'SHORT', 'CLOSE'][action-1]}}, "
-                            f"confidence={{confidence:.3f}} < threshold={thresholds[action-1]}, "
-                            f"uncertainty={{uncertainty:.3f}} > max_sigma_threshold={cfg.backtest.ensemble_max_sigma}"
+                            f": REJECTED {['LONG', 'SHORT', 'CLOSE'][action-1]}, "
+                            f"confidence={confidence:.3f} < threshold={thresholds[action-1]}, "
+                            f"uncertainty={uncertainty:.3f} > max_sigma_threshold={cfg.backtest.ensemble_max_sigma}"
                         )
                         action = 0
 
@@ -405,18 +460,27 @@
     metrics = result.finalize()
     logging.info("\n[Final Metrics]:")
     for name_result, value in metrics.items():
-        logging.info(f": {{name_result:>23s}} = {{value}}")
+        logging.info(f": {name_result:>23s} = {value}")
 
     if cfg.backtest.plot_backtest_balance_curve:
         result.plot_balance(os.path.join(cfg.paths.plot_dir, "backtest_balance_curve.png"))
 
     return metrics
 
 
 if __name__ == "__main__":
-    config_path = sys.argv[1] if len(sys.argv) > 1 else None
-    model_path_arg = sys.argv[2] if len(sys.argv) > 2 else None
+    # --- Modified to load cfg_mod and pass it to run_backtest
+    config_path = sys.argv[1] if len(sys.argv) > 1 else None
+    model_path_arg = sys.argv[2] if len(sys.argv) > 2 else None
 
-    cfg = load_config(config_path) if config_path else default_cfg
-    run_backtest(cfg=cfg, model_path_override=model_path_arg)
+    import importlib.util
+    if config_path:
+        spec = importlib.util.spec_from_file_location("experiment_cfg", config_path)
+        cfg_mod = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(cfg_mod)
+        cfg = cfg_mod.cfg
+    else:
+        cfg_mod = None
+        cfg = default_cfg
+
+    run_backtest(cfg=cfg, cfg_mod=cfg_mod, model_path_override=model_path_arg)
```