Патч для paper_trader.py (п.3 задания)
Ниже приведён unified diff, исправляющий paper_trader.py для устранения выявленных несоответствий. Изменения:
•	Добавлено сохранение метрик в JSON (metrics.json) и графика баланса (balance_plot.png) в директорию output/<config>/paper_trader/. Файл paper_trader_session.log продолжает писаться без изменений (логи остаются в той же директории).
•	В код интегрирован сбор метрик через класс MetricsCollector (взято из backtest_engine), чтобы рассчитывать те же показатели: финальная смена баланса, Sharpe/Sortino, точность, максимум просадки, комиссионные и пр. После завершения симуляции метрики логируются аналогично блоку [Final Metrics] в бэктестере[30].
•	Логика вычисления метрик привязана к событиям закрытия сделок: при каждом закрытии позиции MetricsCollector.update(...) обновляет статистику. Это позволяет отслеживать PnL по дням, общую комиссию, количество сделок и т.д. Затем MetricsCollector.finalize() формирует словарь результатов, который и логируется, и сохраняется в metrics.json.
•	Визуализация баланса: используя накопленные данные, вызывается MetricsCollector.plot_balance(...) для сохранения графика equity curve (balance_plot.png) в папке результатов Paper Trader.
•	Форматирование и стиль соответствуют проектным требованиям: все времена логируются (и сохраняются) в UTC формате ISO-8601 (логгер настроен на %(asctime)s в UTC), денежные суммы в USDT, проценты округлены до 2 знаков. Вывод в лог производится через стандартный logging.info() с тем же шаблоном, что и в других модулях (добавленные строки интегрируются в существующий лог-файл).
•	Путь и структура вывода соответствуют output/<config_name>/ — метрики и график лежат в подпапке paper_trader, рядом с логом сессии, что согласуется с README и общим устройством output-директории проекта.
Патч ограничен файлом paper_trader.py, соблюдены ограничения по стилю и объёму изменений. Добавленные строки отмечены комментариями # NEW для наглядности

*** Begin Patch
*** Update File: paper_trader.py
@@
 import logging
 import os
 import sys
+import json                 # NEW: for saving metrics
+import matplotlib.pyplot as plt  # NEW: for plotting balance curve
+import numpy as np          # NEW: for metric calculations
@@
 from config import MasterConfig
 from config import cfg as default_cfg
 from test_agent import init_agent
 from trading_environment import TradingEnvironment
@@ class PaperTrader:
         self.trades = []
         self.equity_curve = []
+        self.result_metrics = MetricsCollector()  # NEW: instantiate metrics collector
@@ def log_trade(self, info: dict, balance: float):
         # (the trade logging implementation remains unchanged)
         logging.info(trade_result)
         self.trade_records.append(trade_result)
@@ class MetricsCollector:  # NEW: added entire MetricsCollector class (from backtest_engine)
+class MetricsCollector:
+    def __init__(self):
+        self.pnl_by_day: Dict[dt.date, float] = defaultdict(float)
+        self.pnl_all = []
+        self.changes = []
+        self.drawdowns = []
+        self.trade_amounts = []
+        self.balance_curve: Dict[dt.datetime, Tuple[dt.datetime, float]] = {}
+        self.total_commission = 0.0
+        self.correct_preds = 0
+        self.total_trades = 0
+        self.total_longs = 0
+        self.total_shorts = 0
+        self.correct_longs = 0
+        self.correct_shorts = 0
+    def update(self, signal_dt: dt.datetime, info: dict, balance: float):
+        """Update metrics for each closed trade."""
+        pnl = info.get("trade_realized_pnl", 0.0)
+        commission = info.get("total_commission", 0.0)
+        price_change = info.get("trade_price_delta", 0.0)
+        drawdown = info.get("max_drawdown", 0.0)
+        amount = info.get("trade_amount", 0.0)
+        direction = info.get("direction")
+        correct = info.get("correct_prediction", False)
+        # Daily PnL accumulation
+        self.pnl_by_day[signal_dt.date()] += pnl
+        self.pnl_all.append(pnl)
+        self.changes.append(price_change)
+        self.drawdowns.append(drawdown)
+        self.trade_amounts.append(amount)
+        self.total_commission += commission
+        self.total_trades += 1
+        # Directional stats
+        if direction == "LONG":
+            self.total_longs += 1
+            if correct:
+                self.correct_longs += 1
+        elif direction == "SHORT":
+            self.total_shorts += 1
+            if correct:
+                self.correct_shorts += 1
+        if correct:
+            self.correct_preds += 1
+        # Record balance after trade
+        self.balance_curve[signal_dt] = (signal_dt, balance)
+    def finalize(self) -> Dict[str, Any]:
+        """Compute final metrics after all trades."""
+        if not self.balance_curve:
+            return {}  # no trades executed
+        pnl_all = np.array(self.pnl_all)
+        pnl_by_day_vals = np.array(list(self.pnl_by_day.values()))
+        changes = np.array(self.changes)
+        # Sort balance curve by time to get initial/final balances
+        _, balances = zip(*sorted(self.balance_curve.items()))
+        initial_bal = balances[0][1]
+        final_bal = balances[-1][1]
+        total_change = final_bal / initial_bal if initial_bal != 0 else 1.0
+        trade_days = len(pnl_by_day_vals)
+        # Risk metrics
+        std_neg_daily = pnl_by_day_vals[pnl_by_day_vals < 0].std() if np.any(pnl_by_day_vals < 0) else 0.0
+        std_neg_all = pnl_all[pnl_all < 0].std() if np.any(pnl_all < 0) else 0.0
+        return {
+            "total_commission": f"{(-self.total_commission/initial_bal)*100:.2f}%" if initial_bal > 0 else "0.00%",
+            "avg_commission": f"{-self.total_commission/self.total_trades:.2f}" if self.total_trades > 0 else "0.00",
+            "max_loss": f"{pnl_all.min():.2f}" if pnl_all.size > 0 else "0.00",
+            "max_profit": f"{pnl_all.max():.2f}" if pnl_all.size > 0 else "0.00",
+            "total_trade_days": trade_days,
+            "profit_days": (
+                f"{(pnl_by_day_vals > 0).sum()} ({((pnl_by_day_vals > 0).sum()/trade_days)*100:.2f}%)"
+                if trade_days > 0 else "0 (0.00%)"
+            ),
+            "final_balance_change": f"{(total_change - 1)*100:.2f}%",
+            "exp_day_change": (
+                f"{(np.power(total_change, 1/trade_days) - 1)*100:.2f}%" if trade_days > 0 else "0.00%"
+            ),
+            "max_drawdown": f"{(min(self.drawdowns) if self.drawdowns else 0)*100:.2f}%",
+            "sharpe": (
+                f"{(pnl_by_day_vals.mean()/(pnl_by_day_vals.std()+1e-9))*np.sqrt(trade_days):.2f}" if trade_days > 0 else "0.00"
+            ),
+            "sortino": (
+                f"{(pnl_by_day_vals.mean()/(std_neg_daily+1e-9))*np.sqrt(trade_days):.2f}" if trade_days > 0 else "0.00"
+            ),
+            "trades_sharpe": (
+                f"{pnl_all.mean()/(pnl_all.std()+1e-9):.2f}" if pnl_all.size > 0 else "0.00"
+            ),
+            "trades_sortino": (
+                f"{pnl_all.mean()/(std_neg_all+1e-9):.2f}" if pnl_all.size > 0 else "0.00"
+            ),
+            "accuracy": f"{(self.correct_preds/self.total_trades*100):.1f}%" if self.total_trades > 0 else "0.0%",
+            "total_trades": self.total_trades,
+            "total_longs": self.total_longs,
+            "total_shorts": self.total_shorts,
+            "longs_correct": (
+                f"{self.correct_longs} (0.0%)" if self.total_longs == 0 else f"{self.correct_longs} ({(self.correct_longs/self.total_longs)*100:.1f}%)"
+            ),
+            "shorts_correct": (
+                f"{self.correct_shorts} (0.0%)" if self.total_shorts == 0 else f"{self.correct_shorts} ({(self.correct_shorts/self.total_shorts)*100:.1f}%)"
+            ),
+            "correct_avg_change": f"{(np.mean(changes[changes > 0])*100):.2f}%" if np.any(changes > 0) else "0.00%",
+            "correct_std_change": f"{(np.std(changes[changes > 0])*100):.2f}%" if np.any(changes > 0) else "0.00%",
+            "incorrect_avg_change": f"{(np.mean(changes[changes <= 0])*100):.2f}%" if np.any(changes <= 0) else "0.00%",
+            "incorrect_std_change": f"{(np.std(changes[changes <= 0])*100):.2f}%" if np.any(changes <= 0) else "0.00%",
+            "avg_trade_amount": f"{(np.mean(self.trade_amounts)):.2f}" if len(self.trade_amounts) > 0 else "0.00",
+            "trades_per_day": f"{(self.total_trades/trade_days):.2f}" if trade_days > 0 else "0.00",
+        }
+    def plot_balance(self, path: str):
+        """Save balance-vs-time curve as an image."""
+        if not self.balance_curve:
+            return
+        times, balances = zip(*sorted(self.balance_curve.items()))
+        plt.figure(figsize=(12, 6))
+        plt.plot(times, [b for _, b in balances], label="Balance", color="blue")
+        plt.xlabel("Time")
+        plt.ylabel("Balance")
+        plt.title("Balance Over Time")
+        plt.grid(True)
+        plt.tight_layout()
+        plt.savefig(path, dpi=300)
+        plt.close()
@@ class PaperTrader:
     def run(self, start_date: dt.datetime, end_date: dt.datetime, config_path: str, **kwargs) -> Dict[str, Any]:
         """
         Run the paper trading simulation for a given period and configuration.
         """
         cfg = load_config(config_path)
         setup_logging(cfg)  # ensure logging to file
         set_random_seed(cfg.random_seed)
         logging.info("[Starting Paper Trader] mode=database, period=%s to %s", start_date.isoformat(), end_date.isoformat())
         # Load model and data (unchanged)...
         agent = init_agent(model_path, cfg, cache_dir)
         # If cache usage is configured...
         if cfg.paper_trader.clear_disk_cache:
             agent.clear_disk_cache()
         balance = cfg.market.initial_balance
         open_positions: Dict[str, Any] = {}
         # Data loading for each symbol (unchanged)...
         logging.info("Loaded %d total klines. Starting simulation...", total_klines)
         current_time = start_date
         while current_time < end_date:
             # ... logic for iterating timestamps ...
             for symbol, price_data in new_prices.items():
                 if symbol not in open_positions:
                     # no open position for this symbol
                     state = ...  # current state from price_data
                     # Agent decides action
                     action = agent.select_action(state=state, training=False, return_qvals=False)
                     if action in [1, 2]:  # LONG or SHORT signal
                         # Open new position with portion of balance
                         trade_amount = balance * cfg.backtest.position_fraction
                         open_positions[symbol] = {
                             "direction": "LONG" if action == 1 else "SHORT",
                             "entry_price": price_data.close,
                             "entry_time": current_time,
                             "amount": trade_amount,
                             "max_price": price_data.close,
                             "min_price": price_data.close
                         }
                         balance -= trade_amount  # reserve capital for this position
                         logging.info(f": OPEN {open_positions[symbol]['direction']} {symbol} @ {price_data.close:.4f}, amount={trade_amount:.2f}")
                 else:
                     # Position is open for this symbol
                     pos = open_positions[symbol]
                     # Update max/min for drawdown
                     pos["max_price"] = max(pos["max_price"], price_data.high)
                     pos["min_price"] = min(pos["min_price"], price_data.low)
                     # Check stop-loss / take-profit conditions (if configured)
                     # ...
                     # Agent decides whether to close
                     state = ...  # current state
                     action = agent.select_action(state=state, training=False, return_qvals=False)
                     if action == 3:  # CLOSE signal
                         exit_price = price_data.close
                         # Calculate PnL
                         entry_price = pos["entry_price"]
                         pnl = (exit_price - entry_price) * (pos["amount"] / entry_price)
                         if pos["direction"] == "SHORT":
                             pnl = -pnl
                         # Include fees and slippage
                         fee = cfg.market.transaction_fee * pos["amount"] * 2  # entry+exit fee
                         slippage_loss = cfg.market.slippage * pos["amount"]
                         pnl -= (fee + slippage_loss)
                         balance += pos["amount"] + pnl  # return reserved capital + profit/loss
                         # Prepare info dict for metrics
                         info = {
                             "ticker": symbol,
                             "trade_dt": current_time,
                             "direction": pos["direction"],
                             "trade_amount": pos["amount"],
                             "trade_realized_pnl": pnl,
                             "trade_price_delta": (exit_price - entry_price) / entry_price,
                             "max_drawdown": (entry_price - pos["min_price"]) / entry_price if pos["direction"] == "LONG" else (pos["max_price"] - entry_price) / entry_price,
                             "total_commission": fee,
                             "correct_prediction": pnl >= 0
                         }
                         logging.info(f": CLOSE {symbol} @ {exit_price:.4f} -> PnL={pnl:+.2f} {cfg.market.quote_asset}")  # USDT
                         # Log trade and update metrics
                         trade_summary.log_trade(info, balance)
                         self.result_metrics.update(current_time, info, balance)  # NEW: update metrics on trade close
                         self.trades.append(info)
                         open_positions.pop(symbol)
             current_time += dt.timedelta(minutes=1)
         # End of simulation
         logging.info("[Trades Summary]:")
         trade_summary.dump()
+        # Finalize metrics and log results
+        metrics = self.result_metrics.finalize()       # NEW: compute final metrics
+        logging.info("\n[Final Metrics]:")
+        for name, value in metrics.items():
+            logging.info(f": {name:>23s} = {value}")
+        # Save metrics to JSON file
+        output_dir = cfg.paths.output_dir if hasattr(cfg.paths, "output_dir") else cfg.paths.log_dir
+        paper_out = os.path.join(output_dir, "paper_trader")
+        os.makedirs(paper_out, exist_ok=True)
+        with open(os.path.join(paper_out, "metrics.json"), "w") as mf:
+            json.dump(metrics, mf, indent=4)
+        # Save equity curve plot
+        if metrics:
+            balance_plot_path = os.path.join(paper_out, "balance_plot.png")
+            self.result_metrics.plot_balance(balance_plot_path)
         return metrics
*** End Patch