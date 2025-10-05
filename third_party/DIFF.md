*** Begin Patch
*** Update File: third_party/rl-trading-binance/backtest_continuous.py
@@
-    metrics = result.finalize()
-    # Old verbose templated block (risk of missing f-prefix) replaced below
-    logger.info("[Final Metrics]:")
-    for k, v in metrics.items():
-        logger.info(f"{k:>22} = {v}")
+    metrics = result.finalize()
+    # Stable, uniform logging of metrics (pre-formatted values only):
+    logger.info("[Final Metrics]:")
+    order = [
+        "total_commission", "avg_commission",
+        "max_loss", "max_profit",
+        "total_trade_days", "profit_days",
+        "final_balance_change", "exp_day_change",
+        "max_drawdown", "sharpe", "sortino",
+        "trades_sharpe", "trades_sortino",
+        "accuracy", "total_trades", "total_longs", "total_shorts",
+        "longs_correct", "shorts_correct",
+        "correct_avg_change", "correct_std_change",
+        "incorrect_avg_change", "incorrect_std_change",
+        "avg_trade_amount", "trades_per_day",
+    ]
+    for key in order:
+        if key in metrics:
+            logger.info(f"{key:>22} = {metrics[key]}")
*** End Patch
