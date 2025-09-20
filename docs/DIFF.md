diff --git a/src/prosperous_bot/futures_rebalance_backtester.py b/src/prosperous_bot/futures_rebalance_backtester.py
index 69327d9..0000000 100644
--- a/src/prosperous_bot/futures_rebalance_backtester.py
+++ b/src/prosperous_bot/futures_rebalance_backtester.py
@@ -12,4 +12,4 @@
     price = row['close']
-    last_price = None # Store the last price
-    last_price = price # Update last_price in each iteration
+    last_price = None # Последняя цена
+    last_price = price # Обновление последней цены
@@ -29,2 +29,2 @@
-                    if pos['direction'] == 1: # Adding to an existing long position
+                    if pos['direction'] == 1: # Увеличение длинной позиции
@@ -33,2 +33,2 @@
-                    elif pos['direction'] == -1: # Buying to close an existing short position
+                    elif pos['direction'] == -1: # Покупка для закрытия шорт-позиции
@@ -51,2 +51,2 @@
-                else: # No existing position, so this 'buy' opens a new long position
+                else: # Открытие новой длинной позиции
@@ -55,2 +55,2 @@
-                if key in open_positions: # Selling against an existing position
+                if key in open_positions: # Продажа по существующей позиции
@@ -57,2 +57,2 @@
-                    if pos['direction'] == 1: # Selling to close an existing long position
+                    if pos['direction'] == 1: # Продажа для закрытия длинной позиции
@@ -75,2 +75,2 @@
-                    elif pos['direction'] == -1: # Adding to an existing short position
+                    elif pos['direction'] == -1: # Увеличение шорт-позиции
@@ -79,2 +79,2 @@
-                else: # No existing position, so this 'sell' opens a new short position
+                else: # Открытие новой шорт-позиции
@@ -84,2 +84,2 @@
-    if force_close_open_positions and open_positions and last_price is not None: # Ensure there was data
+    if force_close_open_positions and open_positions and last_price is not None: # Убеждаемся, что данные непусты
@@ -86,1 +86,1 @@
-        'exit_price': last_price, # Close at the last known price
+        'exit_price': last_price, # Закрытие по последней известной цене
@@ -93,1 +93,1 @@
-        'status': 'force_closed' # Add a status for these trades
+        'status': 'force_closed' # Статус принудительного закрытия позиции
@@ -113,1 +113,1 @@
-from .logging_config import configure_root # This will be adjusted by hand later if patch fails
+from .logging_config import configure_root # Настройка корневого логгера
@@ -119,1 +119,1 @@
-# Basic logging configuration
+# Базовая конфигурация логирования
@@ -153,1 +153,1 @@
-        # Standardize 'timestamp' column to UTC.
+        # Стандартизация меток времени в UTC.
@@ -159,1 +159,1 @@
-        # Drop invalid rows
+        # Удаляем некорректные строки
@@ -162,1 +162,1 @@
-        df = df_signals.dropna(subset=['timestamp']) # Changed df_signals to df
+        df = df_signals.dropna(subset=['timestamp']) # Удаляем строки с некорректными метками времени
@@ -169,1 +169,1 @@
-        df_signals = df_signals[['timestamp', 'signal']].sort_values(by='timestamp', ascending=True)
+        df_signals = df_signals[['timestamp', 'signal']].sort_values(by='timestamp', ascending=True) # Оставляем только нужные столбцы и сортируем
@@ -245,1 +245,1 @@
-# --- START OF REPLACEMENT FUNCTION ---
+# --- Начало функции run_backtest ---
@@ -247,1 +247,1 @@
-    # deep-copy → подстановка плейс-холдеров не изменит исходный dict
+    # Глубокое копирование: замена плейсхолдеров не изменит исходный словарь
@@ -250,1 +250,1 @@
-    #  Neutral “ideal-conditions” run: отключаем ЛЮБЫЕ фильтры на
+    #  Нейтральный «идеальный» прогон: отключаем любые фильтры на
