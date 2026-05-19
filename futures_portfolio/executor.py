"""Исполнение ордеров на Binance Futures (Hedge Mode)."""
import logging
import asyncio
from typing import Dict, Optional, Tuple, Any, List, TYPE_CHECKING
from decimal import Decimal, ROUND_HALF_EVEN, getcontext

if TYPE_CHECKING:
    from connector import BinanceConnector

logger = logging.getLogger(__name__)

# Set decimal precision and rounding mode globally for financial calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

class PortfolioExecutor:
    def __init__(self, connector: 'BinanceConnector', base_ticker: str = "BTCUSDT", max_orders_per_second: int = 10):
        self.connector = connector
        self.base_ticker = base_ticker
        self.semaphore = asyncio.Semaphore(max_orders_per_second)

    def calculate_order_size(self, target_share: float, current_value: float, total_value: float, spot_price: float) -> Decimal:
        """Расчёт размера ордера в контрактах."""
        dec_target_share = Decimal(str(target_share))
        dec_current_value = Decimal(str(current_value))
        dec_total_value = Decimal(str(total_value))
        dec_spot_price = Decimal(str(spot_price))
        
        target_value = dec_target_share * dec_total_value
        delta_value = target_value - dec_current_value
        order_qty = delta_value / dec_spot_price
        return order_qty

    def round_quantity(self, qty: Decimal, step_size: Decimal) -> Decimal:
        """
        Умное округление: использует шаг биржи, но ограничивает точность до 3 знаков (0.001).
        Это позволяет торговать дорогими монетами (TAO, BTC) и при этом избегать 'пыли'.
        """
        if step_size <= 0:
            return qty.quantize(Decimal('1.000'), rounding="ROUND_FLOOR").normalize()

        # Используем шаг биржи. Если он слишком мелкий (например, 0.000001), 
        # укрупняем его до 0.001 для чистоты учета.
        effective_step = max(step_size, Decimal('0.001'))
        
        # Округляем ВНИЗ до ближайшего разрешенного шага
        rounded = (qty / effective_step).quantize(Decimal('1'), rounding="ROUND_FLOOR") * effective_step
        return rounded.normalize()

    async def execute_market_order(self, symbol: str, qty: Any, side: str, step_size: Any = Decimal('0'), reduce_only: bool = False, position_side: str = "BOTH", min_notional: Any = Decimal('6.0'), price: Any = Decimal('0')) -> Dict:
        """
        Отправка рыночного ордера на Binance Futures с полной точностью Decimal.
        """
        # Ensure Decimal for all numeric inputs to prevent float errors
        qty = Decimal(str(qty))
        step_size = Decimal(str(step_size))
        min_notional = Decimal(str(min_notional))
        price = Decimal(str(price))

        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": "Округлилось до нуля"}

        # Проверка минимальной стоимости (Notional Value)
        try:
            if price <= 0:
                prices = await self.connector.get_futures_prices([symbol])
                price = Decimal(str(prices.get(symbol, 0.0)))

            if price > 0 and (abs(qty) * price) < min_notional:
                msg = f"Order too small: {float(abs(qty) * price):.2f} USDT < {float(min_notional)} USDT. Skipping."
                logger.info(msg)
                return {"status": "SKIPPED", "message": msg}
        except Exception as e:
            logger.warning(f"Could not verify notional value: {e}")

        try:
            # Конвертируем в строку для API Binance, чтобы избежать float-погрешностей
            str_qty = "{:f}".format(abs(qty).normalize())

            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": str_qty,  # Передаем строкой! Binance API это отлично переваривает
                "positionSide": position_side
            }
            
            result = await asyncio.to_thread(
                self.connector.futures_client.futures_create_order,
                **params
            )

            # Извлекаем реальный исполненный объем из ответа биржи
            order_id = result.get("orderId")
            executed_qty = Decimal(str(result.get("executedQty", "0.0")))
            avg_price = Decimal(str(result.get("avgPrice", "0.0")))
            
            # [POLLING] Если Бинанс вернул 0 (бывает при задержках в ядре)
            if executed_qty == 0 and order_id:
                logger.info(f"Market order {order_id} initial fill is 0. Polling status...")
                for _ in range(5):
                    await asyncio.sleep(0.3)
                    status = await self.connector.get_order_status(symbol, order_id)
                    executed_qty = Decimal(str(status.get("executedQty", "0.0")))
                    avg_price = Decimal(str(status.get("avgPrice", "0.0")))
                    if executed_qty > 0: break

            if avg_price == 0 and executed_qty > 0:
                avg_price = price

            logger.info(f"Order EXECUTED on Binance: {side} {str_qty} {symbol} ({position_side}) | Fact Qty: {executed_qty}")

            if executed_qty == 0:
                return {"status": "ERROR", "message": f"Binance executed 0.0 contracts for order {order_id}", "result": result}

            # --- REAL TRADE DATA SYNC ---
            # Fetch actual trades to get real realized PnL and commission
            real_pnl = Decimal('0.0')
            real_commission = Decimal('0.0')
            try:
                # Даем бирже время на расчет трейдов
                await asyncio.sleep(0.5)
                trades = await self.connector.get_order_trades(symbol, order_id)
                for t in trades:
                    real_pnl += Decimal(str(t.get("realizedPnl", "0.0")))
                    real_commission += Decimal(str(t.get("commission", "0.0")))
                logger.info(f"Realized data for {symbol} Order {order_id}: PnL={float(real_pnl):+.4f}, Commission={float(real_commission):.4f}")
            except Exception as e:
                logger.warning(f"Could not fetch trade details for {symbol} {order_id}: {e}")

            return {
                "status": "SUCCESS",
                "result": result,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "realized_pnl": real_pnl,
                "commission": real_commission
            }
        except Exception as e:
            import traceback
            logger.error(f"Order FAILED for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return {"status": "ERROR", "message": str(e) if str(e) else f"Unknown error of type {type(e).__name__}"}

    async def execute_limit_with_fallback(self, symbol: str, qty: Any, side: str,
                                           step_size: Any = Decimal('0'), reduce_only: bool = False,
                                           position_side: str = "BOTH",
                                           offset_pct: Any = Decimal('0.2'),
                                           timeout_sec: int = 30,
                                           min_notional: Any = Decimal('6.0'),
                                           price: Any = Decimal('0')) -> Dict:
        """
        Limit + Fallback с проверкой минимальной стоимости с использованием Decimal.
        """
        qty = Decimal(str(qty))
        step_size = Decimal(str(step_size))
        offset_pct = Decimal(str(offset_pct))
        min_notional = Decimal(str(min_notional))
        price = Decimal(str(price))

        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": "Округлилось до нуля"}

        try:
            # 1. Получаем mid-price и проверяем стоимость
            order_book = await self.connector.get_order_book(symbol, limit=5)
            best_bid = Decimal(str(order_book["bids"][0][0]))
            best_ask = Decimal(str(order_book["asks"][0][0]))
            mid_price = (best_bid + best_ask) / 2

            if (abs(qty) * mid_price) < min_notional:
                msg = f"Limit order too small: {float(abs(qty) * mid_price):.2f} USDT < {float(min_notional)} USDT. Skipping."
                logger.info(msg)
                return {"status": "SKIPPED", "message": msg}

            # 2. Рассчитываем цену лимитки (выгоднее mid-price)
            if side == "SELL":
                limit_price = mid_price * (1 + offset_pct / 100)
                if limit_price > best_ask: limit_price = best_ask
            else:
                limit_price = mid_price * (1 - offset_pct / 100)
                if limit_price < best_bid: limit_price = best_bid

            # Логирование для статистики
            expected_improvement_pct = abs(limit_price - mid_price) / mid_price * 100
            logger.info(f"Limit order: {side} {float(qty)} {symbol} @ {float(limit_price):.6f} (mid={float(mid_price):.6f}, expected_gain={float(expected_improvement_pct):.3f}%)")

            # 3. Выставляем POST-ONLY лимитку (гарантия maker-комиссии 0.02%)
            params = {
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "price": float(limit_price),
                "position_side": position_side
            }
            
            order = await self.connector.place_limit_maker_order(**params)

            order_id = order["orderId"]
            logger.info(f"Limit order placed: {order_id}")

            # 4. Ждём исполнения
            filled_qty = Decimal('0.0')
            avg_fill_price = Decimal('0.0')

            for _ in range(timeout_sec):
                await asyncio.sleep(1)
                status = await self.connector.get_order_status(symbol, order_id)
                filled_qty = Decimal(str(status.get("executedQty", 0)))
                avg_fill_price = Decimal(str(status.get("avgPrice", 0))) if status.get("avgPrice") else avg_fill_price

                if status["status"] == "FILLED":
                    # --- REAL DATA SYNC ---
                    real_pnl = Decimal('0.0')
                    real_commission = Decimal('0.0')
                    try:
                        await asyncio.sleep(0.5)
                        trades = await self.connector.get_order_trades(symbol, order_id)
                        for t in trades:
                            real_pnl += Decimal(str(t.get("realizedPnl", "0.0")))
                            real_commission += Decimal(str(t.get("commission", "0.0")))
                    except: pass

                    return {
                        "status": "SUCCESS_LIMIT",
                        "executed_qty": filled_qty,
                        "avg_price": avg_fill_price,
                        "realized_pnl": real_pnl,
                        "commission": real_commission,
                        "order_type": "LIMIT_MAKER"
                    }

            # 5. Timeout — отменяем и добиваем market
            logger.warning(f"Limit order timeout ({timeout_sec}s). Cancelling and fallback to market.")
            await self.connector.cancel_order(symbol, order_id)

            # Добиваем остаток market-ордером
            remaining_qty = qty - filled_qty
            market_pnl = Decimal('0.0')
            market_comm = Decimal('0.0')
            total_executed = filled_qty
            
            if remaining_qty > 0:
                if (abs(remaining_qty) * mid_price) >= min_notional:
                    market_result = await self.execute_market_order(
                        symbol=symbol, qty=remaining_qty, side=side, step_size=Decimal('0'),
                        position_side=position_side, min_notional=min_notional, price=mid_price
                    )
                    if market_result["status"] == "SUCCESS":
                        total_executed += Decimal(str(market_result["executed_qty"]))
                        market_pnl = Decimal(str(market_result["realized_pnl"]))
                        market_comm = Decimal(str(market_result["commission"]))

            limit_pnl = Decimal('0.0')
            limit_comm = Decimal('0.0')
            if filled_qty > 0:
                try:
                    await asyncio.sleep(0.3)
                    trades = await self.connector.get_order_trades(symbol, order_id)
                    for t in trades:
                        limit_pnl += Decimal(str(t.get("realizedPnl", "0.0")))
                        limit_comm += Decimal(str(t.get("commission", "0.0")))
                except: pass

            return {
                "status": "SUCCESS_FALLBACK",
                "executed_qty": total_executed,
                "realized_pnl": limit_pnl + market_pnl,
                "commission": limit_comm + market_comm
            }

        except Exception as e:
            logger.error(f"Limit+Fallback FAILED: {e}")
            return await self.execute_market_order(
                symbol=symbol, qty=qty, side=side, step_size=step_size,
                reduce_only=reduce_only, position_side=position_side,
                min_notional=min_notional, price=price
            )


    def get_limit_order_params(self, config: dict) -> Tuple[bool, Decimal, int]:
        """
        Извлечение параметров лимитных ордеров из конфига.
        Возвращает: (enabled, offset_pct, timeout_sec)
        """
        enabled = config.get("limit_order_enabled", False)
        offset_pct = Decimal(str(config.get("limit_offset_pct", 0.2)))
        timeout_sec = config.get("limit_timeout_sec", 30)
        return enabled, offset_pct, timeout_sec

    async def execute_rebalance(self, action: Dict[str, Any], price: float, step_size: float, limit_order: bool = True, portfolio_cfg: Optional[Dict[str, Any]] = None) -> bool:
        """
        Совместимость со старым кодом: выполнение одного действия ребалансировки.
        """
        symbol = action["symbol"]

        base_symbol = action.get("base_symbol")
        pos_side = action.get("position_side")

        if not base_symbol or not pos_side:
            symbol_parts = symbol.split('_')
            base_symbol = symbol_parts[0]
            pos_side = symbol_parts[1] if len(symbol_parts) > 1 else "LONG"

        diff_usdt = Decimal(str(action["diff_usdt"]))
        dec_price = Decimal(str(price))
        side = ("BUY" if diff_usdt > 0 else "SELL") if pos_side == "LONG" else ("SELL" if diff_usdt > 0 else "BUY")
        order_qty = abs(diff_usdt / dec_price)
        reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")

        min_notional = Decimal(str(portfolio_cfg.get("min_notional_usdt", 6.0))) if portfolio_cfg else Decimal('6.0')
        dec_step_size = Decimal(str(step_size))

        if limit_order:
            limit_enabled, limit_offset, limit_timeout = self.get_limit_order_params(portfolio_cfg or {})
            res = await self.execute_limit_with_fallback(
                symbol=base_symbol, qty=order_qty, side=side, step_size=dec_step_size,
                reduce_only=reduce_only, position_side=pos_side, offset_pct=limit_offset,
                timeout_sec=limit_timeout, min_notional=min_notional, price=dec_price
            )
        else:
            res = await self.execute_market_order(
                symbol=base_symbol, qty=order_qty, side=side, step_size=dec_step_size,
                reduce_only=reduce_only, position_side=pos_side, min_notional=min_notional, price=dec_price
            )

        return res["status"] in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]

    async def _execute_single_action(self, action: Dict[str, Any], price: float, paper_mode: bool, portfolio_cfg: Dict[str, Any], step_sizes: Dict[str, float], paper_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Внутренний метод для выполнения одного действия (для gather)."""
        if action["type"] == "VIRTUAL_ORDER":
            diff_usdt = action["diff_usdt"]
            side = "BUY" if diff_usdt > 0 else "SELL"
            return {
                "type": "VIRTUAL_ORDER",
                "status": "SUCCESS",
                "diff_usdt": diff_usdt,
                "side": side
            }

        symbol = action["symbol"]

        base_symbol = action.get("base_symbol")
        pos_side = action.get("position_side")

        if not base_symbol or not pos_side:
            symbol_parts = symbol.split('_')
            base_symbol = symbol_parts[0]
            pos_side = symbol_parts[1] if len(symbol_parts) > 1 else "LONG"

        diff_usdt = Decimal(str(action["diff_usdt"]))
        dec_price = Decimal(str(price))
        side = ("BUY" if diff_usdt > 0 else "SELL") if pos_side == "LONG" else ("SELL" if diff_usdt > 0 else "BUY")
        order_qty = abs(diff_usdt / dec_price)
        reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")
        step_size = Decimal(str(step_sizes.get(base_symbol, 0.0))) if step_sizes else Decimal('0.0')
        min_notional = Decimal(str(portfolio_cfg.get("min_notional_usdt", 6.0)))

        if paper_mode:
            qty_rounded = self.round_quantity(order_qty, step_size)
            order_value = qty_rounded * dec_price
            
            if order_value < min_notional:
                return {
                    "type": pos_side,
                    "symbol": symbol,
                    "side": side,
                    "qty": 0.0,
                    "status": "SKIPPED",
                    "message": f"Order too small: {float(order_value):.2f} USDT < {float(min_notional)} USDT"
                }

            if qty_rounded <= 0:
                return {
                    "type": pos_side,
                    "symbol": symbol,
                    "side": side,
                    "qty": 0.0,
                    "status": "SKIPPED",
                    "message": "Округлено до нуля"
                }
            commission = order_value * Decimal('0.0004')

            trade_pnl = Decimal('0.0')
            if paper_state:
                entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
                old_entry = Decimal(str(paper_state.get(entry_key, price)))
                if reduce_only:
                    if pos_side == "LONG":
                        trade_pnl = qty_rounded * (dec_price - old_entry)
                    else:
                        trade_pnl = qty_rounded * (old_entry - dec_price)

            return {
                "type": pos_side,
                "symbol": symbol,
                "side": side,
                "qty": float(qty_rounded),
                "price": float(dec_price),
                "status": "SUCCESS",
                "trade_pnl": float(trade_pnl),
                "commission": float(commission),
                "reduce_only": reduce_only
            }
        else:
            # Real mode
            async with self.semaphore:
                limit_enabled, limit_offset, limit_timeout = self.get_limit_order_params(portfolio_cfg or {})
                if limit_enabled:
                    res = await self.execute_limit_with_fallback(
                        symbol=base_symbol, qty=order_qty, side=side, step_size=step_size,
                        reduce_only=reduce_only, position_side=pos_side, offset_pct=limit_offset,
                        timeout_sec=limit_timeout, min_notional=min_notional, price=dec_price
                    )
                else:
                    # Передаем Decimal напрямую без кастинга во float!
                    res = await self.execute_market_order(
                        symbol=base_symbol, qty=order_qty, side=side, step_size=step_size,
                        reduce_only=reduce_only, position_side=pos_side, min_notional=min_notional, price=dec_price
                    )

                if res["status"] in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]:
                    # Берем строго то, что исполнила биржа
                    executed_qty = res.get("executed_qty", Decimal('0.0'))
                    avg_price = res.get("avg_price", dec_price)
                    trade_pnl = res.get("realized_pnl", Decimal('0.0'))
                    commission = res.get("commission", Decimal('0.0'))

                    if executed_qty == 0:
                        return {
                            "type": pos_side, "symbol": symbol, "side": side, "qty": 0.0,
                            "status": "ERROR", "message": "Fact executed qty is zero"
                        }

                    return {
                        "type": pos_side,
                        "symbol": symbol,
                        "side": side,
                        "qty": float(executed_qty),
                        "price": float(avg_price),
                        "status": "SUCCESS",
                        "reduce_only": reduce_only,
                        "trade_pnl": float(trade_pnl),
                        "commission": float(commission)
                    }
                
                return {
                    "type": pos_side,
                    "symbol": symbol,
                    "side": side,
                    "qty": float(res.get("executed_qty", Decimal('0.0'))),
                    "price": float(res.get("avg_price", dec_price)),
                    "status": res["status"],
                    "exec_res": res,
                    "reduce_only": reduce_only,
                    "trade_pnl": float(trade_pnl),
                    "commission": float(commission)
                }

    async def execute_actions(self, actions: List[Dict[str, Any]], price: float, paper_mode: bool = True, portfolio_cfg: Optional[Dict[str, Any]] = None, step_sizes: Optional[Dict[str, float]] = None, paper_state: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Векторизованное (конкурентное) выполнение действий по ребалансировке с принудительным Market-only исполнением.
        SURPLUS-FIRST: Reductions (SELLs) are executed before Expansions (BUYs).
        """
        if not actions:
            return []

        if portfolio_cfg:
            portfolio_cfg["limit_order_enabled"] = False

        # Surplus-First Doctrine: Execute SELLs (reductions) before BUYs (expansions)
        reductions = [a for a in actions if a.get("is_reduction")]
        expansions = [a for a in actions if not a.get("is_reduction")]

        # Mapping for result tracking to maintain original order
        results_map = {}

        for group in [reductions, expansions]:
            if not group:
                continue

            # Map action IDs to their results
            group_action_ids = [id(a) for a in group]
            tasks = [
                self._execute_single_action(action, price, paper_mode, portfolio_cfg or {}, step_sizes or {}, paper_state)
                for action in group
            ]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for action_id, r in zip(group_action_ids, group_results):
                if isinstance(r, Exception):
                    logger.error(f"Action execution failed with exception: {r}")
                    results_map[action_id] = {"status": "ERROR", "message": str(r)}
                else:
                    results_map[action_id] = r

        # Reconstruct results in original order
        final_results = [results_map[id(a)] for a in actions]
        success_count = sum(1 for r in final_results if r.get("status") in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"])

        if len(actions) > 0:
            logger.info(f"Executed {success_count}/{len(actions)} actions in two phases (Surplus-First).")

        return final_results
