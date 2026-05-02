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
        """Математически корректное округление количества до шага лота с использованием Decimal."""
        if step_size <= 0:
            return qty

        # Strictly mathematically correct rounding for arbitrary steps using Decimal
        return (qty / step_size).quantize(Decimal('1'), rounding=ROUND_HALF_EVEN) * step_size

    async def execute_market_order(self, symbol: str, qty: float, side: str, step_size: float = 0.0, reduce_only: bool = False, position_side: str = "BOTH", min_notional: float = 6.0, price: float = 0.0) -> Dict:
        """
        Отправка рыночного ордера на Binance Futures.
        """
        dec_qty = Decimal(str(qty))
        dec_step_size = Decimal(str(step_size))
        dec_min_notional = Decimal(str(min_notional))
        dec_price = Decimal(str(price))

        if dec_qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if dec_step_size > 0:
            dec_qty = self.round_quantity(dec_qty, dec_step_size)
            if dec_qty == 0:
                return {"status": "NO_ORDER", "message": f"Округлилось до нуля"}

        # Проверка минимальной стоимости (Notional Value)
        try:
            if dec_price <= 0:
                # Получаем текущую цену только если она не передана
                prices = await self.connector.get_futures_prices([symbol])
                dec_price = Decimal(str(prices.get(symbol, 0.0)))

            if dec_price > 0 and (abs(dec_qty) * dec_price) < dec_min_notional:
                msg = f"Order too small: {float(abs(dec_qty) * dec_price):.2f} USDT < {float(dec_min_notional)} USDT. Skipping."
                logger.info(msg)
                return {"status": "SKIPPED", "message": msg}
        except Exception as e:
            logger.warning(f"Could not verify notional value: {e}")

        try:
            # Обязательно передаем positionSide для Hedge Mode
            result = await asyncio.to_thread(
                self.connector.futures_client.futures_create_order,
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=float(abs(dec_qty)),
                reduceOnly=reduce_only,
                positionSide=position_side
            )
            logger.info(f"Order: {side} {float(abs(dec_qty))} {symbol} ({position_side})")
            return {"status": "SUCCESS", "result": result}
        except Exception as e:
            logger.error(f"Order FAILED: {e}")
            return {"status": "ERROR", "message": str(e)}

    async def execute_limit_with_fallback(self, symbol: str, qty: float, side: str,
                                           step_size: float = 0.0, reduce_only: bool = False,
                                           position_side: str = "BOTH",
                                           offset_pct: float = 0.2,
                                           timeout_sec: int = 30,
                                           min_notional: float = 6.0,
                                           price: float = 0.0) -> Dict:
        """
        Limit + Fallback с проверкой минимальной стоимости.
        """
        dec_qty = Decimal(str(qty))
        dec_step_size = Decimal(str(step_size))
        dec_min_notional = Decimal(str(min_notional))
        dec_offset_pct = Decimal(str(offset_pct))

        if dec_qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if dec_step_size > 0:
            dec_qty = self.round_quantity(dec_qty, dec_step_size)
            if dec_qty == 0:
                return {"status": "NO_ORDER", "message": f"Округлилось до нуля"}

        try:
            # 1. Получаем mid-price и проверяем стоимость
            order_book = await self.connector.get_order_book(symbol, limit=5)
            best_bid = Decimal(str(order_book["bids"][0][0]))
            best_ask = Decimal(str(order_book["asks"][0][0]))
            mid_price = (best_bid + best_ask) / 2

            if (abs(dec_qty) * mid_price) < dec_min_notional:
                msg = f"Limit order too small: {float(abs(dec_qty) * mid_price):.2f} USDT < {float(dec_min_notional)} USDT. Skipping."
                logger.info(msg)
                return {"status": "SKIPPED", "message": msg}

            # 2. Рассчитываем цену лимитки (выгоднее mid-price)
            if side == "SELL":
                limit_price = mid_price * (1 + dec_offset_pct / 100)
                if limit_price > best_ask: limit_price = best_ask
            else:
                limit_price = mid_price * (1 - dec_offset_pct / 100)
                if limit_price < best_bid: limit_price = best_bid

            # Логирование для статистики
            expected_improvement_pct = abs(limit_price - mid_price) / mid_price * 100
            logger.info(f"Limit order: {side} {float(dec_qty)} {symbol} @ {float(limit_price):.6f} (mid={float(mid_price):.6f}, expected_gain={float(expected_improvement_pct):.3f}%)")

            # 3. Выставляем POST-ONLY лимитку (гарантия maker-комиссии 0.02%)
            order = await self.connector.place_limit_maker_order(
                symbol=symbol,
                side=side,
                qty=float(dec_qty),
                price=float(limit_price),
                position_side=position_side,
                reduce_only=reduce_only
            )

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
                    # Расчёт выигрыша в цене
                    price_improvement_pct = (avg_fill_price - mid_price) / mid_price * 100 if side == "SELL" else (mid_price - avg_fill_price) / mid_price * 100
                    notional = filled_qty * avg_fill_price
                    profit_usdt = notional * (price_improvement_pct / 100)

                    logger.info(f"Limit FILLED: {float(filled_qty)} @ {float(avg_fill_price):.6f} | mid={float(mid_price):.6f} | gain={float(price_improvement_pct):+.3f}% ({float(profit_usdt):+.2f} USDT)")
                    return {
                        "status": "SUCCESS_LIMIT",
                        "filled_qty": float(filled_qty),
                        "avg_price": float(avg_fill_price),
                        "mid_price": float(mid_price),
                        "price_improvement_pct": float(price_improvement_pct),
                        "profit_usdt": float(profit_usdt),
                        "order_type": "LIMIT_MAKER"
                    }

                # Частичное исполнение — обновляем qty для fallback
                if filled_qty > 0:
                    dec_qty = dec_qty - filled_qty

            # 5. Timeout — отменяем и добиваем market
            logger.warning(f"Limit order timeout ({timeout_sec}s). Cancelling and fallback to market.")
            await self.connector.cancel_order(symbol, order_id)

            # Если было частичное исполнение — логируем
            if filled_qty > 0:
                price_improvement_pct = (avg_fill_price - mid_price) / mid_price * 100 if side == "SELL" else (mid_price - avg_fill_price) / mid_price * 100
                logger.info(f"Partial fill: {float(filled_qty)} @ {float(avg_fill_price):.6f} | gain={float(price_improvement_pct):+.3f}%")

            # Добиваем остаток market-ордером
            if dec_qty > 0:
                # CRITICAL: Re-verify notional value for the remaining snippet
                if (abs(dec_qty) * mid_price) < dec_min_notional:
                    logger.warning(f"Fallback snippet too small ({float(abs(dec_qty) * mid_price):.2f} < {float(dec_min_notional)}). Discarding remainder.")
                    return {"status": "SUCCESS_PARTIAL", "filled_qty": float(filled_qty), "message": "Remainder dropped due to min_notional"}
                market_result = await self.execute_market_order(
                    symbol=symbol,
                    qty=float(dec_qty),
                    side=side,
                    step_size=0,  # Уже округлено
                    reduce_only=reduce_only,
                    position_side=position_side
                )
                return {
                    "status": "SUCCESS_FALLBACK",
                    "limit_filled_qty": float(filled_qty),
                    "limit_avg_price": float(avg_fill_price),
                    "limit_mid_price": float(mid_price),
                    "limit_price_improvement_pct": float(price_improvement_pct) if filled_qty > 0 else 0,
                    "market_result": market_result,
                    "order_type": "LIMIT_THEN_MARKET"
                }

        except Exception as e:
            logger.error(f"Limit+Fallback FAILED: {e}")
            # Fallback на market в случае ошибки
            try:
                market_result = await self.execute_market_order(
                    symbol=symbol,
                    qty=float(dec_qty),
                    side=side,
                    step_size=float(dec_step_size),
                    reduce_only=reduce_only,
                    position_side=position_side
                )
                return {"status": "ERROR_FALLBACK", "market_result": market_result, "error": str(e)}
            except Exception as e2:
                logger.error(f"Market fallback also FAILED: {e2}")
                return {"status": "ERROR", "message": f"{e}; Fallback: {e2}"}

    def get_limit_order_params(self, config: dict) -> Tuple[bool, float, int]:
        """
        Извлечение параметров лимитных ордеров из конфига.
        Возвращает: (enabled, offset_pct, timeout_sec)
        """
        enabled = config.get("limit_order_enabled", False)
        offset_pct = config.get("limit_offset_pct", 0.2)
        timeout_sec = config.get("limit_timeout_sec", 30)
        return enabled, offset_pct, timeout_sec

    async def execute_rebalance(self, action: Dict[str, Any], price: float, step_size: float, limit_order: bool = True, portfolio_cfg: Optional[Dict[str, Any]] = None) -> bool:
        """
        Совместимость со старым кодом: выполнение одного действия ребалансировки.
        """
        symbol = action["symbol"]

        # Оптимизация: используем предварительно разложенные поля, если они есть
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

        min_notional = portfolio_cfg.get("min_notional_usdt", 6.0) if portfolio_cfg else 6.0

        if limit_order:
            limit_enabled, limit_offset, limit_timeout = self.get_limit_order_params(portfolio_cfg or {})
            res = await self.execute_limit_with_fallback(
                symbol=base_symbol, qty=float(order_qty), side=side, step_size=step_size,
                reduce_only=reduce_only, position_side=pos_side, offset_pct=limit_offset,
                timeout_sec=limit_timeout, min_notional=min_notional, price=price
            )
        else:
            res = await self.execute_market_order(
                symbol=base_symbol, qty=float(order_qty), side=side, step_size=step_size,
                reduce_only=reduce_only, position_side=pos_side, min_notional=min_notional, price=price
            )

        return res["status"] in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"]

    async def _execute_single_action(self, action: Dict[str, Any], price: float, paper_mode: bool, portfolio_cfg: Dict[str, Any], step_sizes: Dict[str, float], paper_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Внутренний метод для выполнения одного действия (для gather)."""
        if action["type"] == "VIRTUAL_RESET":
            return {"type": "VIRTUAL_RESET", "status": "SUCCESS"}

        symbol = action["symbol"]

        # Оптимизация: используем предварительно разложенные поля, если они есть
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
            if qty_rounded <= 0:
                return {
                    "type": pos_side,
                    "symbol": symbol,
                    "side": side,
                    "qty": 0.0,
                    "status": "SKIPPED",
                    "message": "Округлено до нуля"
                }

            order_value = qty_rounded * dec_price
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
                        symbol=base_symbol, qty=float(order_qty), side=side, step_size=float(step_size),
                        reduce_only=reduce_only, position_side=pos_side, offset_pct=limit_offset,
                        timeout_sec=limit_timeout, min_notional=float(min_notional), price=price
                    )
                else:
                    res = await self.execute_market_order(
                        symbol=base_symbol, qty=float(order_qty), side=side, step_size=float(step_size),
                        reduce_only=reduce_only, position_side=pos_side, min_notional=float(min_notional), price=price
                    )

                return {
                    "type": pos_side,
                    "symbol": symbol,
                    "side": side,
                    "qty": float(order_qty),
                    "price": float(dec_price),
                    "status": res["status"],
                    "exec_res": res,
                    "reduce_only": reduce_only,
                    "trade_pnl": 0.0 # В реальном режиме PnL определяется биржей
                }

    async def execute_actions(self, actions: List[Dict[str, Any]], price: float, paper_mode: bool = True, portfolio_cfg: Optional[Dict[str, Any]] = None, step_sizes: Optional[Dict[str, float]] = None, paper_state: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Векторизованное (конкурентное) выполнение действий по ребалансировке.
        """
        if not actions:
            return []

        tasks = [
            self._execute_single_action(action, price, paper_mode, portfolio_cfg or {}, step_sizes or {}, paper_state)
            for action in actions
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Action execution failed with exception: {r}")
                final_results.append({"status": "ERROR", "message": str(r)})
            else:
                final_results.append(r)

        success_count = sum(1 for r in final_results if r.get("status") in ["SUCCESS", "SUCCESS_LIMIT", "SUCCESS_FALLBACK"])
        if len(actions) > 0:
            logger.info(f"Executed {success_count}/{len(actions)} actions concurrently.")

        return final_results
