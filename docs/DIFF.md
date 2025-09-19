--- src/prosperous_bot/futures_rebalance_backtester.py
+++ src/prosperous_bot/futures_rebalance_backtester.py
@@ -293,12 +293,17 @@ def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_
     taker_commission_rate = _get_commission_rate(params, maker=False)
     use_maker_fees_in_backtest = bool(params.get('use_maker_fees_in_backtest', False))
     slippage_percent = params.get('slippage_percent', params.get('slippage_percentage', 0.0005))
-    circuit_breaker_threshold_percent = params.get('circuit_breaker_threshold_percent', 0.10) 
-    margin_usage_safe_mode_enter_threshold = params.get('margin_usage_safe_mode_enter_threshold', 0.70) 
-    margin_usage_safe_mode_exit_threshold = params.get('margin_usage_safe_mode_exit_threshold', 0.50)
-    safe_mode_target_weights = params.get('safe_mode_target_weights', target_weights_normal)
+    circuit_breaker_cfg = params.get('circuit_breaker_config', {})
+    circuit_breaker_threshold_percent = circuit_breaker_cfg.get('threshold_percentage', 0.0)
+    safe_mode_cfg = params.get('safe_mode_config', {})
+    margin_usage_safe_mode_enter_threshold = safe_mode_cfg.get('entry_threshold', 0.0)
+    margin_usage_safe_mode_exit_threshold = safe_mode_cfg.get('exit_threshold', 0.0)
+    safe_mode_target_weights = safe_mode_cfg.get('target_weights_safe', target_weights_normal)
     min_rebalance_interval_minutes = params.get('min_rebalance_interval_minutes', 0)

     main_asset_symbol = params.get('main_asset_symbol', 'BTC')
@@ -342,6 +347,13 @@ def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_
         os.makedirs(actual_reports_dir, exist_ok=True)
+        # Настройка логирования в файл в папке отчётов
+        log_file_path = os.path.join(actual_reports_dir, "backtest.log")
+        file_handler = logging.FileHandler(log_file_path)
+        file_handler.setLevel(logging.INFO)
+        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
+        file_handler.setFormatter(formatter)
+        logging.getLogger().addHandler(file_handler)

     df_market_original = load_data(data_path) # Keep original for plotting price
     if df_market_original is None or df_market_original.empty:
@@ -878,6 +891,9 @@ def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_
     df_trades = pd.DataFrame(trades_list)
+    # Округляем денежные параметры сделок до 2 знаков
+    df_trades[['quantity_quote','entry_price','exit_price','commission_quote','slippage_quote','pnl_gross_quote','pnl_net_quote']] = \
+        df_trades[['quantity_quote','entry_price','exit_price','commission_quote','slippage_quote','pnl_gross_quote','pnl_net_quote']].round(2)

     # ---------- PERFORMANCE METRICS ----------
     def compute_metrics(df_eq: pd.DataFrame, trades: list[dict], initial_nav: float, ann_factor: int = 252):
@@ -970,6 +986,12 @@ def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_
     metrics["num_blocked_trades"] = len(blocked_trades_list)

+    # Округляем итоговые метрики: деньги до 2 знаков, проценты до 3 знаков
+    for k, v in metrics.items():
+        if isinstance(v, float):
+            if k.endswith('_usdt'):
+                metrics[k] = round(v, 2)
+            elif k.endswith('_percent'):
+                metrics[k] = round(v, 3)

     if generate_reports and actual_reports_dir:
         logging.info(f"Generating reports in {actual_reports_dir}...")
@@ -1073,6 +1095,9 @@ def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_
             fig.write_html(equity_html_path)
             logging.info(f"Enhanced equity curve saved to {equity_html_path}")
+
+            # Округляем кривую equity до 2 знаков перед сохранением
+            df_equity['portfolio_value_usdt'] = df_equity['portfolio_value_usdt'].round(2)
             equity_csv_path = os.path.join(actual_reports_dir, "equity.csv")
             df_equity.to_csv(equity_csv_path, index=False)
             logging.info(f"Equity curve data saved to {equity_csv_path}")
