diff --git a/src/prosperous_bot/futures_rebalance_backtester.py b/src/prosperous_bot/futures_rebalance_backtester.py
--- a/src/prosperous_bot/futures_rebalance_backtester.py
+++ b/src/prosperous_bot/futures_rebalance_backtester.py
@@
-def calculate_portfolio_value(usdt_balance, 
-                              btc_long_value_usdt, btc_short_value_usdt):
+def calculate_portfolio_value(usdt_balance,
+                              btc_long_value_usdt, btc_short_value_usdt):
@@
-    total_value = usdt_balance + btc_long_value_usdt + btc_short_value_usdt
-    return total_value
+    return usdt_balance + btc_long_value_usdt + btc_short_value_usdt
@@
-def record_trade(timestamp, asset_type, action, quantity_asset, quantity_quote, market_price, 
-                 commission_usdt, slippage_usdt, pnl_net_quote, trades_list):
-    """Записывает информацию о сделке, симулированной в процессе бэктеста.
-    - quantity_asset: количество базового актива (например, BTC) в сделке.
-    - quantity_quote: стоимость сделки в USDT до учета комиссий и проскальзывания.
-    - market_price: цена актива в момент совершения сделки.
-    - commission_usdt: комиссия за сделку в USDT.
-    - slippage_usdt: стоимость проскальзывания в USDT.
-    - pnl_net_quote: чистая прибыль/убыток по сделке в USDT после учета комиссий и проскальзывания.
-    """
+def record_trade(timestamp, asset_type, action, quantity_asset, quantity_quote, market_price,
+                 commission_usdt, slippage_usdt, trades_list):
+    """Записывает сделку бэктеста (фьючерсы).
+    Параметры:
+        timestamp: метка времени сделки (UTC).
+        asset_type: ключ актива (например, BTC_PERP_LONG / BTC_PERP_SHORT).
+        action: BUY/SELL (направление изменения позиции).
+        quantity_asset: количество базового актива (в единицах базового актива, до плеча).
+        quantity_quote: номинал сделки в USDT (до комиссии и проскальзывания).
+        market_price: рыночная цена при фиксации сделки.
+        commission_usdt: комиссия сделки в USDT.
+        slippage_usdt: проскальзывание сделки в USDT.
+        trades_list: список для накопления сделок.
+    Примечания:
+        Для сделок ребалансировки «моментный» PnL не фиксируется. Брутто-PnL = 0.0,
+        нетто-PnL учитывает только издержки: -(комиссия + проскальзывание).
+    """
     trade = {
         "timestamp_open": timestamp, 
         "timestamp_close": timestamp, 
         "asset_type": asset_type,
         "action": action, 
         "quantity_asset": quantity_asset, 
         "quantity_quote": quantity_quote, 
         "entry_price": market_price, 
         "exit_price": market_price, 
         "commission_quote": commission_usdt,
         "slippage_quote": slippage_usdt, 
-        "pnl_gross_quote": pnl_net_quote, 
-        "pnl_net_quote": pnl_net_quote - commission_usdt, 
+        "pnl_gross_quote": 0.0,
+        "pnl_net_quote": -(commission_usdt + slippage_usdt),
     }
     trades_list.append(trade)
     logging.info(
-        f"  СДЕЛКА: {action} {quantity_asset:.6f} {asset_type} @ MktPx {market_price:.2f}, "
-        f"Стоимость: {quantity_quote:.2f}, Комиссия: {commission_usdt:.2f}, Стоимость проскальзывания: {slippage_usdt:.2f}, "
-        f"Чистый PnL сделки: {(pnl_net_quote - commission_usdt):.2f}"
+        f"  СДЕЛКА: {action} {quantity_asset:.6f} {asset_type} по {market_price:.2f}, "
+        f"номинал: {quantity_quote:.2f}, комиссия: {commission_usdt:.2f}, проскальз.: {slippage_usdt:.2f}, "
+        f"нетто PnL сделки: {-(commission_usdt + slippage_usdt):.2f}"
     )
@@
-# --- Начало функции run_backtest ---
-def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_reports=None):
+# --- Начало функции run_backtest ---
+def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_reports=None):
+    """Запускает бэктест дельта-нейтральной стратегии фьючерсной ребалансировки.
+    Входные параметры:
+        params_dict (dict): настройки из unified_config (секция backtest_settings).
+        data_path (str): путь к CSV с историей рынка.
+        is_optimizer_call (bool): при True отчёты могут быть упрощены (для оптимизатора).
+        trial_id_for_reports (int|None): id прогона для структурирования отчётов.
+    Алгоритм:
+        1) Загрузка и нормализация данных рынка; привязка сигналов из CSV (merge_asof).
+        2) Главный цикл: пересчёт PnL фьючерсных ног, контроль Safe-Mode/АВ, проверка порогов.
+        3) Формирование ребаланс-ордеров, учёт комиссий/проскальзывания, запись сделок.
+        4) (Опц.) Симуляция исполнения для аналитики (simulate_rebalance) и сохранение отчётов.
+    Выход:
+        dict: ключевые метрики (PnL, Sharpe, PF, Win-Rate, MaxDD), статус, путь к отчётам.
+    """
@@
-    spot_asset_key = f"{main_asset_symbol}_SPOT"
     long_asset_key = f"{main_asset_symbol}_PERP_LONG"
     short_asset_key = f"{main_asset_symbol}_PERP_SHORT"
@@
-                if dust_filter_on:
+                if dust_filter_on:
                     min_nominal = params.get("min_order_notional_usdt", 10.0)
                     if abs(usdt_value_to_trade) < min_nominal:
                         continue
@@
-                if asset_key_trade == short_asset_key:
+                if asset_key_trade == short_asset_key:
                     # увеличиваем шорт → SELL, уменьшаем → BUY
                     action_dir = "SELL" if usdt_value_to_trade > 0 else "BUY"
-                else:   # спот / лонг
+                else:   # лонг
                     action_dir = "BUY"  if usdt_value_to_trade > 0 else "SELL"
@@
-                if asset_key_trade == long_asset_key:
+                if asset_key_trade == long_asset_key:
                     order_type = "OPEN_LONG"  if action_dir == "BUY"  else "CLOSE_LONG"
                 elif asset_key_trade == short_asset_key:
                     # OPEN_SHORT ⇔ SELL,   CLOSE_SHORT ⇔ BUY
                     order_type = "OPEN_SHORT" if action_dir == "SELL" else "CLOSE_SHORT"
-                else:                              # спотовая нога
-                    order_type = action_dir            # BUY/SELL
+                else:
+                    # Неизвестный или не-фьючерсный ключ (например, устаревший SPOT/USDT) — пропускаем
+                    continue
@@
-                quantity_asset_traded_final = 0.0
-                realized_pnl_this_spot_trade = 0.0
+                quantity_asset_traded_final = 0.0
                 slippage_cost_this_trade_usdt = abs_usdt_value_of_trade * slippage_percent
@@
-                # Определяем количество для orders_by_step, должно быть в терминах актива
-                qty_for_orders = 0
-                if current_price > 0: # Избегаем деления на ноль, если цена почему-то ноль
-                    if asset_key_trade == spot_asset_key:
-                        qty_for_orders = abs(usdt_value_to_trade) / current_price
-                    elif asset_key_trade == long_asset_key or asset_key_trade == short_asset_key:
-                        # Для активов с плечом кол-во также должно быть в базовом активе для simulate_rebalance
-                        qty_for_orders = abs(usdt_value_to_trade) / current_price
-                        # Примечание: simulate_rebalance применяет плечо, поэтому qty здесь - это кол-во актива до плеча
+                # Количество для simulate_rebalance (в базовом активе, до плеча)
+                qty_for_orders = 0
+                if current_price > 0 and asset_key_trade in (long_asset_key, short_asset_key):
+                    qty_for_orders = abs(usdt_value_to_trade) / current_price
@@
-                # Передаем action_dir (BUY/SELL) как 'action' для записи о сделке
-                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
-                             abs_usdt_value_of_trade, current_price, commission_usdt,
-                             slippage_cost_this_trade_usdt, realized_pnl_this_spot_trade, trades_list)
+                # Записываем сделку (PnL моментно не фиксируем — только издержки)
+                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
+                             abs_usdt_value_of_trade, current_price, commission_usdt,
+                             slippage_cost_this_trade_usdt, trades_list)
@@
-                    if current_signal == "BUY":
-                        if (asset_key_loop == spot_asset_key or asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt < 0:
+                    if current_signal == "BUY":
+                        if (asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt < 0:
                             trade_blocked_by_signal = True
                         elif asset_key_loop == short_asset_key and original_proposed_adjustment_usdt > 0:
                             trade_blocked_by_signal = True
-                    elif current_signal == "SELL":
-                        if asset_key_loop == short_asset_key and original_proposed_adjustment_usdt < 0:
+                    elif current_signal == "SELL":
+                        if asset_key_loop == short_asset_key and original_proposed_adjustment_usdt < 0:
                             trade_blocked_by_signal = True
-                        elif (asset_key_loop == spot_asset_key or asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt > 0:
+                        elif (asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt > 0:
                             trade_blocked_by_signal = True
@@
-                if asset_key_trade == "USDT": continue
+                if asset_key_trade == "USDT":
+                    continue
+                # Пропускаем все ключи, не относящиеся к фьючерсам LONG/SHORT
+                if asset_key_trade not in (long_asset_key, short_asset_key):
+                    logging.info("Пропущен не-фьючерсный ключ веса: %s", asset_key_trade)
+                    continue
@@
-            asset_colors = {
-                spot_asset_key: {'BUY': 'rgba(0,128,0,0.9)', 'SELL': 'rgba(255,0,0,0.9)'},
-                long_asset_key: {'BUY': 'rgba(0,0,255,0.7)', 'SELL': 'rgba(255,140,0,0.7)'},
-                short_asset_key: {'BUY': 'rgba(128,0,128,0.7)', 'SELL': 'rgba(165,42,42,0.7)'}
-            }
-            asset_symbols = {
-                spot_asset_key: {'BUY': 'triangle-up', 'SELL': 'triangle-down'},
-                long_asset_key: {'BUY': 'circle', 'SELL': 'circle-open'},
-                short_asset_key: {'BUY': 'star', 'SELL': 'star-open'}
-            }
+            asset_colors = {
+                long_asset_key: {'BUY': 'rgba(0,0,255,0.7)', 'SELL': 'rgba(255,140,0,0.7)'},
+                short_asset_key: {'BUY': 'rgba(128,0,128,0.7)', 'SELL': 'rgba(165,42,42,0.7)'}
+            }
+            asset_symbols = {
+                long_asset_key: {'BUY': 'circle', 'SELL': 'circle-open'},
+                short_asset_key: {'BUY': 'star', 'SELL': 'star-open'}
+            }
@@
-            if not df_trades.empty:
-                for asset_name_key_plot in [spot_asset_key, long_asset_key, short_asset_key]:
+            if not df_trades.empty:
+                for asset_name_key_plot in [long_asset_key, short_asset_key]:
                     for action_str_plot in ['BUY', 'SELL']:
                         trades_to_plot = df_trades[
                             (df_trades['action'] == action_str_plot) &
                             (df_trades['asset_type'] == asset_name_key_plot)
                         ]
                         if not trades_to_plot.empty:
