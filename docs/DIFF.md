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
