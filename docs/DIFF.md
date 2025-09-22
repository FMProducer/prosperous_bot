--- futures_rebalance_backtester.py
+++ futures_rebalance_backtester.py
@@ -80,7 +80,7 @@
                 else: # No existing position, so this 'sell' opens a new short position
                     pos = open_positions[key]
                     entry = pos['entry_price']
-    # Force-closure of any remaining open positions at the end of the data
+    # Принудительное закрытие всех оставшихся открытых позиций в конце исторических данных
     if force_close_open_positions and open_positions and last_price is not None: # Ensure there was data
         for key, pos in list(open_positions.items()): # Use list to allow modification
             pnl = (last_price - pos['entry_price']) * pos['qty'] * pos['direction'] * leverage
@@ -134,14 +134,14 @@
     """Loads and processes signal data from a CSV file."""
     logging.info(f"Attempting to load signal data from {signal_csv_path}...")
     try:
         df_signals = pd.read_csv(signal_csv_path)
         if df_signals.empty:
             logging.warning(f"Signal file found at {signal_csv_path} but it is empty.")
             return None
 
-    """Loads historical market data from CSV."""
+    """Загружает исторические рыночные данные из CSV-файла."""
     logging.info(f"Loading data from {csv_path}...")
     try:
         df = pd.read_csv(csv_path)
@@ -200,25 +200,20 @@
 def calculate_portfolio_value(usdt_balance, btc_spot_qty, 
                               btc_long_value_usdt, btc_short_value_usdt, 
                               current_btc_price):
-    """
-    Calculates the current total portfolio value in USDT.
-    For Part 1, btc_long_value_usdt and btc_short_value_usdt are the current market values
-    of the capital allocated to these leveraged strategies, including their P&L.
-    """
+    """Вычисляет текущую стоимость портфеля в USDT (без учета спотового актива).
+    Для этого суммируется баланс USDT и значения фьючерсных позиций.
+    """
-    value_spot_btc = btc_spot_qty * current_btc_price
-    total_value = usdt_balance + value_spot_btc + btc_long_value_usdt + btc_short_value_usdt
+    total_value = usdt_balance + btc_long_value_usdt + btc_short_value_usdt
     return total_value
diff
Копировать код
--- futures_rebalance_backtester.py
+++ futures_rebalance_backtester.py
@@ -212,16 +212,20 @@
 def record_trade(timestamp, asset_type, action, quantity_asset, quantity_quote, market_price, 
                  commission_usdt, slippage_usdt, pnl_net_quote, trades_list):
-    """
-    Records a simulated trade.
-    - quantity_asset: For BTC_SPOT, this is BTC. For leveraged, this is the USDT value being allocated/deallocated.
-    - quantity_quote: USDT value of the trade *before* commission & slippage.
-    - market_price: Price of BTC at the time of trade decision.
-    - commission_usdt: Commission paid in USDT.
-    - slippage_usdt: Cost of slippage in USDT.
-    - pnl_net_quote: Net PnL of this trade in USDT (primarily for SPOT, after costs).
-    - realized_pnl_spot_usdt: The portion of pnl_net_quote that is from realized SPOT gains/losses.
-    """
-    trade = {
+    """Записывает информацию о сделке, симулированной в процессе бэктеста.
+    - quantity_asset: количество базового актива (например, BTC) в сделке.
+    - quantity_quote: стоимость сделки в USDT до учета комиссий и проскальзывания.
+    - market_price: цена актива в момент совершения сделки.
+    - commission_usdt: комиссия за сделку в USDT.
+    - slippage_usdt: стоимость проскальзывания в USDT.
+    - pnl_net_quote: чистая прибыль/убыток по сделке в USDT после учета комиссий и проскальзывания.
+    """
+    trade = {
         "timestamp_open": timestamp, 
         "timestamp_close": timestamp, 
         "asset_type": asset_type,
@@ -228,8 +232,8 @@
         "entry_price": market_price, 
         "exit_price": market_price, 
         "commission_quote": commission_usdt,
         "slippage_quote": slippage_usdt, 
-        "pnl_gross_quote": realized_pnl_spot_usdt, 
-        "pnl_net_quote": realized_pnl_spot_usdt - commission_usdt, 
+        "pnl_gross_quote": pnl_net_quote, 
+        "pnl_net_quote": pnl_net_quote - commission_usdt, 
     }
     trades_list.append(trade)
     logging.info(
         f"  TRADE: {action} {quantity_asset:.6f} {asset_type} @ MktPx {market_price:.2f}, "
@@ -240,7 +244,7 @@
         f"Val: {quantity_quote:.2f}, Comm: {commission_usdt:.2f}, SlipCost: {slippage_usdt:.2f}, "
-        f"NetPnL_Trade: {(realized_pnl_spot_usdt - commission_usdt):.2f}"
+        f"NetPnL_Trade: {(pnl_net_quote - commission_usdt):.2f}"
     )
diff
Копировать код
--- futures_rebalance_backtester.py
+++ futures_rebalance_backtester.py
@@ -480,7 +480,7 @@
         "total_trades": 0,
         "sharpe_ratio": 0.0,
         "sortino_ratio": 0.0,
         "max_drawdown_percent": 0.0,
         "profit_factor": 0.0,
         "win_rate_percent": 0.0,
         "output_dir": output_dir,
     }
@@ -483,11 +483,8 @@
     portfolio = {
-        'usdt_balance': initial_portfolio_value_usdt, 'btc_spot_qty': 0.0,
-        'btc_spot_lots': [], 'btc_long_value_usdt': 0.0, 'btc_short_value_usdt': 0.0,
+        'usdt_balance': initial_portfolio_value_usdt,
+        'btc_long_value_usdt': 0.0, 'btc_short_value_usdt': 0.0,
         'prev_btc_price': None, 'total_commissions_usdt': 0.0, 'total_slippage_usdt': 0.0,
         'current_operational_mode': 'NORMAL_MODE', 'num_circuit_breaker_triggers': 0,
         'num_safe_mode_entries': 0, 'time_steps_in_safe_mode': 0,
         'last_rebalance_attempt_timestamp': None,
     }
diff
Копировать код
--- futures_rebalance_backtester.py
+++ futures_rebalance_backtester.py
@@ -647,14 +644,12 @@
                 "USDT": portfolio['usdt_balance'] / total_portfolio_value if total_portfolio_value else 1,
-                spot_asset_key: (portfolio['btc_spot_qty'] * current_price) / total_portfolio_value if total_portfolio_value else 0,
                 long_asset_key: portfolio['btc_long_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
                 short_asset_key: portfolio['btc_short_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
             }
             for key in active_target_weights:
                 if key not in current_weights:
                     current_weights[key] = 0.0
 
@@ -669,7 +664,8 @@
                 portfolio['btc_long_value_usdt'] += long_pnl
                 portfolio['btc_short_value_usdt'] += short_pnl
 
-        total_portfolio_value = calculate_portfolio_value(
-            portfolio['usdt_balance'], portfolio['btc_spot_qty'],
+        total_portfolio_value = calculate_portfolio_value(
+            portfolio['usdt_balance'], portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'],
             current_price)
 
         nav = total_portfolio_value
@@ -679,13 +675,16 @@
             for asset_key_loop, target_w_loop in active_target_weights.items():
                 # scale PERP notional by leverage so that pnl ~ leverage
                 if asset_key_loop in (long_asset_key, short_asset_key):
                     target_value_usdt = target_w_loop * total_portfolio_value
                 else:
                     target_value_usdt = target_w_loop * total_portfolio_value
                 current_value_usdt = 0
+                if asset_key_loop == long_asset_key: current_value_usdt = portfolio['btc_long_value_usdt']
+                elif asset_key_loop == short_asset_key: current_value_usdt = portfolio['btc_short_value_usdt']
+                elif asset_key_loop == "USDT": current_value_usdt = portfolio['usdt_balance']
-                elif asset_key_loop == long_asset_key: current_value_usdt = portfolio['btc_long_value_usdt']
-                elif asset_key_loop == short_asset_key: current_value_usdt = portfolio['btc_short_value_usdt']
-                elif asset_key_loop == "USDT": current_value_usdt = portfolio['usdt_balance']
-                
+                
                 adjustment_usdt = target_value_usdt - current_value_usdt
 
                 if apply_signal_logic:
diff
Копировать код
--- futures_rebalance_backtester.py
+++ futures_rebalance_backtester.py
@@ -776,13 +777,6 @@
                 quantity_asset_traded_final = 0.0
                 realized_pnl_this_spot_trade = 0.0
                 slippage_cost_this_trade_usdt = abs_usdt_value_of_trade * slippage_percent
-
-                # ---------- SPOT BTC ----------
-                if asset_key_trade == spot_asset_key:
-                    qty_btc = abs_usdt_value_of_trade / current_price
-                    quantity_asset_traded_final = qty_btc
-                    if order_type == "BUY": # Spot BUY
-                        portfolio["btc_spot_qty"] = portfolio.get("btc_spot_qty", 0.0) + qty_btc
-                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade
-                    else: # Spot SELL
-                        qty_close = min(qty_btc, portfolio.get("btc_spot_qty", 0.0))
-                        portfolio["btc_spot_qty"] -= qty_close
-                        portfolio["usdt_balance"] += qty_close * current_price
-                        realized_pnl_this_spot_trade = (current_price - portfolio.get("prev_btc_price", current_price)) * qty_close
-
-                # ---------- PERP LONG ----------
-                elif asset_key_trade == long_asset_key:
+                # ---------- PERP LONG ----------
+                if asset_key_trade == long_asset_key:
                     quantity_asset_traded_final = abs_usdt_value_of_trade # For futures, asset quantity is the quote value
                     if order_type == "OPEN_LONG":
                         portfolio["btc_long_value_usdt"] = portfolio.get("btc_long_value_usdt", 0.0) + abs_usdt_value_of_trade
@@ -843,12 +837,13 @@
                                 last entry price, exit price, quantity, PnL, etc.
                                 May be the current timestamp if entry/exit is instantaneous (as in market orders).
-                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
-                             abs_usdt_value_of_trade, current_price, commission_usdt,
-                             slippage_cost_this_trade_usdt, realized_pnl_this_spot_trade, trades_list,
-                             realized_pnl_spot_usdt=realized_pnl_this_spot_trade)
+                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
+                             abs_usdt_value_of_trade, current_price, commission_usdt,
+                             slippage_cost_this_trade_usdt, realized_pnl_this_spot_trade, trades_list)
diff
Копировать код
--- futures_rebalance_backtester.py
+++ futures_rebalance_backtester.py
@@ -840,7 +837,6 @@
                             'asset_key': asset_key_trade,
                             'side': action_dir,
                             'qty': qty_for_orders
                         })
 
                 # Pass action_dir (BUY/SELL) as the 'action' for the trade record
-                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
+                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
                              abs_usdt_value_of_trade, current_price, commission_usdt,
                              slippage_cost_this_trade_usdt, realized_pnl_this_spot_trade, trades_list)