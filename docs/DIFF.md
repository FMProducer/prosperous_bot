@@ def load_signal_data(signal_csv_path: str) -> pd.DataFrame | None:
-        df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'], utc=True, errors='coerce', format='ISO8601')
+        # Надёжный парсинг ISO-строк без явного format для совместимости версий pandas
+        df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'], utc=True, errors='coerce')
@@
-        if df_signals['timestamp'].dt.tz is None:
+        if df_signals['timestamp'].dt.tz is None:
             logging.info(f"Signal data 'timestamp' column from {signal_csv_path} is tz-naive. Localizing to UTC.")
             df_signals['timestamp'] = df_signals['timestamp'].dt.tz_localize('UTC')
         else:
             logging.info(f"Signal data 'timestamp' column from {signal_csv_path} is already tz-aware ({df_signals['timestamp'].dt.tz}). Converting to UTC.")
             df_signals['timestamp'] = df_signals['timestamp'].dt.tz_convert('UTC')

@@ def run_backtest(...):
-    if df_signals is not None and not df_signals.empty:
-        logging.info("Merging signal data with market data using merge_asof (backward)...")
-        df_market = pd.merge_asof(df_market, df_signals[['timestamp', 'signal']],
-                                  on='timestamp', direction='backward')
+    if df_signals is not None and not df_signals.empty:
+        logging.info("Merging signal data with market data using merge_asof (backward)...")
+        # merge_asof требует сортировку по ключу
+        df_market = df_market.sort_values('timestamp')
+        df_signals = df_signals.sort_values('timestamp')
+        df_market = pd.merge_asof(
+            df_market, df_signals[['timestamp', 'signal']],
+            on='timestamp', direction='backward'
+        )
         df_market['signal'] = df_market['signal'].ffill()
         logging.info("Signal data merged. 'signal' column is now available in market data.")

@@ def run_backtest(...):
-        log_file_path = os.path.join(actual_reports_dir, "backtest.log")
-        file_handler = logging.FileHandler(log_file_path)
-        file_handler.setLevel(logging.INFO)
-        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
-        file_handler.setFormatter(formatter)
-        logging.getLogger().addHandler(file_handler)
+        log_file_path = os.path.join(actual_reports_dir, "backtest.log")
+        root_logger = logging.getLogger()
+        # Не добавляем повторно хендлер тот же файл
+        if not any(
+            isinstance(h, logging.FileHandler)
+            and getattr(h, "baseFilename", None) == os.path.abspath(log_file_path)
+            for h in root_logger.handlers
+        ):
+            file_handler = logging.FileHandler(log_file_path)
+            file_handler.setLevel(logging.INFO)
+            formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
+            file_handler.setFormatter(formatter)
+            root_logger.addHandler(file_handler)

@@ def run_backtest(...):
-    df_trades = pd.DataFrame(trades_list)
-    # Округляем денежные параметры сделок до 2 знаков
-    df_trades[['quantity_quote','entry_price','exit_price','commission_quote','slippage_quote','pnl_gross_quote','pnl_net_quote']] = \
-        df_trades[['quantity_quote','entry_price','exit_price','commission_quote','slippage_quote','pnl_gross_quote','pnl_net_quote']].round(2)
+    df_trades = pd.DataFrame(trades_list)
+    # Округление безопасно, даже если часть колонок отсутствует или нет сделок
+    if not df_trades.empty:
+        cols_to_round = [
+            c for c in (
+                'quantity_quote','entry_price','exit_price',
+                'commission_quote','slippage_quote','pnl_gross_quote','pnl_net_quote'
+            ) if c in df_trades.columns
+        ]
+        if cols_to_round:
+            df_trades[cols_to_round] = df_trades[cols_to_round].round(2)

@@ def run_backtest(...):
-    def compute_metrics(df_eq: pd.DataFrame, trades: list[dict], initial_nav: float, ann_factor: int = 252):
+    def compute_metrics(df_eq: pd.DataFrame, trades: list[dict], initial_nav: float, ann_factor: int = 252):
@@
-        if trades:
-            pnl_list = [t.get("pnl_net_quote", 0.0) for t in trades]
-            wins = [p for p in pnl_list if p > 0]
-            losses = [-p for p in pnl_list if p < 0]
-            out["profit_factor"] = (sum(wins) / sum(losses)) if losses else 0.0
-            out["win_rate_percent"] = (len(wins) / len(pnl_list)) * 100 if pnl_list else 0.0
-        else:
-            out["profit_factor"] = 0.0
-            out["win_rate_percent"] = 0.0
+        if trades:
+            pnl_list = [t.get("pnl_net_quote", 0.0) for t in trades]
+            wins = [p for p in pnl_list if p > 0]
+            losses = [-p for p in pnl_list if p < 0]
+            out["profit_factor"] = (sum(wins) / sum(losses)) if losses else 0.0
+            out["win_rate_percent"] = (len(wins) / max(1, len(pnl_list))) * 100
+        else:
+            out["profit_factor"] = 0.0
+            out["win_rate_percent"] = 0.0
+        # Fallback: если в трейд-логе нет информативных pnl (например, только комиссии),
+        # оценим win-rate и PF по ряду доходностей equity
+        if out.get("profit_factor", 0.0) == 0.0 and out.get("win_rate_percent", 0.0) == 0.0 and not df_eq.empty:
+            rets = df_eq["portfolio_value_usdt"].pct_change().dropna()
+            wins = (rets > 0).sum()
+            losses = (rets < 0).sum()
+            out["win_rate_percent"] = (wins / max(1, wins + losses)) * 100
+            out["profit_factor"] = (
+                rets[rets > 0].sum() / abs(rets[rets < 0].sum())
+            ) if losses else 0.0

