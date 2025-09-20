@@ -246,0 +247,10 @@ def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_reports=None):
+    """
+    Выполняет бэктест дельта-нейтральной стратегии ребаланса фьючерсов.
+    Параметры:
+        params_dict (dict): Настройки стратегии и бэктестера (комиссии, левередж, таргетные веса, пороги и т.д.).
+        data_path (str): Путь к CSV-файлу(ам) исторических рыночных данных.
+        is_optimizer_call (bool): Флаг режима оптимизации (True при вызове из оптимизатора, отчёты не сохраняются).
+        trial_id_for_reports (Optional[int]): Идентификатор прогона для формирования отчётов (используется при оптимизации).
+    Этапы работы:
+        1. Загрузка и предобработка данных рынка.
+        2. Применение сигналов (если включена соответствующая логика).
+        3. Моделирование сделок по целевым весам и ребалансировка портфеля.
+        4. Учет комиссий, проскальзывания (slippage), TP/SL, Safe Mode и Circuit Breaker.
+        5. Сбор метрик (KPI) и формирование equity-кривой и журнала сделок.
+    Возвращает:
+        dict: Результаты бэктеста, включая итоговую стоимость портфеля, PnL, статистические метрики и статус выполнения.
+    """
@@ -333,1 +333,1 @@
-            if not output_dir: # If prefix was empty or just "/"
+            if not output_dir: # Если префикс пустой или равен "/"
@@ -334,1 +334,1 @@
-                output_dir = "reports" # Default to "reports" to be safe for tests
+                output_dir = "reports" # По умолчанию 'reports'
@@ -337,1 +337,1 @@
-            # Existing logic for timestamped/optimizer paths
+            # Логика формирования пути отчётов
@@ -370,2 +370,2 @@
-    logging.info("Report generation is OFF. No reports will be saved.")
+    logging.info("Report generation is OFF. No reports will be saved.")
-    # output_dir remains None as it's not used when reports are off.
+    # output_dir остаётся None, поскольку при отключенных отчётах он не используется.
@@ -374,1 +374,1 @@
-    df_market_original = load_data(data_path) # Keep original for plotting price
+    df_market_original = load_data(data_path) # Оригинальные данные рынка для графика цены
@@ -384,1 +384,1 @@
-                                   "conditional_value_at_risk_cvar_percent", "omega_ratio", "ulcer_index", "skewness", "kurtosis")} # Added more zeroed metrics
+                                   "conditional_value_at_risk_cvar_percent", "omega_ratio", "ulcer_index", "skewness", "kurtosis")} # Добавлены новые метрики с нулевыми значениями
@@ -386,1 +386,1 @@
-            "final_portfolio_value_usdt": initial_portfolio_value_usdt, # Corrected
+            "final_portfolio_value_usdt": initial_portfolio_value_usdt, # Исправлено
@@ -387,1 +387,1 @@
-            "total_net_pnl_usdt": 0.0, # Corrected
+            "total_net_pnl_usdt": 0.0, # Исправлено
@@ -388,1 +388,1 @@
-            "total_net_pnl_percent": 0.0, # Corrected
+            "total_net_pnl_percent": 0.0, # Исправлено
@@ -395,1 +395,1 @@
-    df_market = df_market_original.copy() # Work with a copy for potential modifications
+    df_market = df_market_original.copy() # Работаем с копией данных для изменений
@@ -404,1 +404,1 @@
-    # ── auto-range: если "auto" или дата вне диапазона файла ────────────
+    # ── авто-диапазон: если "auto" или дата вне диапазона файла ────────────
@@ -407,1 +407,1 @@
-    dr = params.setdefault("date_range", {}) # Get or create 'date_range' dict
+    dr = params.setdefault("date_range", {}) # Получаем или создаём словарь 'date_range'
@@ -453,1 +453,1 @@
-            except Exception as e: # More general exception
+            except Exception as e: # Общий перехват исключений
@@ -467,1 +467,1 @@
-    if df_market.empty:                     # graceful-fail for unit-tests
+    if df_market.empty:                     # Корректный выход для unit-тестов
@@ -503,1 +503,1 @@
-        # Return structure consistent with other error returns
+        # Структура ответа при ошибке (как и в других случаях)
@@ -544,1 +544,1 @@
-                     metrics_cb_fail = {key: 0 for key in ["sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_percent"]} # Initialize all expected keys
+                     metrics_cb_fail = {key: 0 for key in ["sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_percent"]} # Инициализируем все необходимые ключи
@@ -548,1 +548,1 @@
-                         "max_drawdown_percent": -100.0, # Or calculate actual if possible
+                         "max_drawdown_percent": -100.0, # или вычислить реальный, если возможно
@@ -551,1 +551,1 @@
-                         **portfolio # Spread existing portfolio state
+                         **portfolio # Добавляем текущее состояние портфеля
