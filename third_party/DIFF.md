diff --git a/third_party/rl-trading-binance/paper_trader.py b/third_party/rl-trading-binance/paper_trader.py
--- a/third_party/rl-trading-binance/paper_trader.py
+++ b/third_party/rl-trading-binance/paper_trader.py
@@ -214,10 +214,20 @@
         if len(self.open_positions) >= self.cfg.backtest.max_parallel_sessions:
             logging.warning("Max parallel sessions reached. Skipping new trade.")
             return
 
         direction = "LONG" if action == 1 else "SHORT"
-        position_size = self.balance * self.cfg.backtest.position_fraction
+        # Reserve capital from free balance (cash) to avoid over-allocation across parallel trades
+        free_cash = self.balance
+        position_size = free_cash * self.cfg.backtest.position_fraction
+        if position_size <= 0:
+            logging.warning("Insufficient free balance to open position.")
+            return
 
         # --- NEW: Initialize risk management state ---
         rm_state = {}
         if self.cfg.backtest.use_risk_management:
             if direction == "LONG":
                 rm_state["trailing_max_price"] = entry_price
             else: # SHORT
                 rm_state["trailing_min_price"] = entry_price
 
         self.open_positions[symbol] = {
             "direction": direction,
             "entry_price": entry_price,
             "entry_time": signal_dt,
             "size": position_size,
             "close_time": signal_dt + dt.timedelta(minutes=self.cfg.seq.agent_session_len),
             "max_price": entry_price,
             "min_price": entry_price,
             **rm_state
         }
+        # Reserve principal immediately (matches backtest-style cash accounting)
+        self.balance -= position_size
+        # Optional: record equity after opening (for more granular plot)
+        self.equity_curve.append({"ts": signal_dt.isoformat(), "balance": self.balance})
         logging.info(
             f"PAPER TRADE OPEN: {direction} {symbol} at {entry_price:.4f} (Size: {position_size:.2f} USDT)"
         )
@@ -260,7 +270,8 @@
             fees = (pos["size"] / pos["entry_price"] * pos["entry_price"] * self.cfg.market.transaction_fee) + \
                    (pos["size"] / pos["entry_price"] * current_price * self.cfg.market.transaction_fee)
             net_pnl = pnl - fees
 
-            self.balance += net_pnl
+            # Release principal and add realized PnL (entry+exit fees already included in net_pnl)
+            self.balance += pos["size"] + net_pnl
 
             # Prepare info dict for metrics
             price_delta = (current_price - pos["entry_price"]) / pos["entry_price"]
             if pos["direction"] == "SHORT":
                 price_delta = -price_delta
