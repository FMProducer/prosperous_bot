"""Исполнение ордеров на Binance Futures (Hedge Mode)."""
import logging
import asyncio
import math
from typing import Dict, Optional, Tuple, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from connector import BinanceConnector

logger = logging.getLogger(__name__)


class PortfolioExecutor:
    def __init__(self, connector: 'BinanceConnector', base_ticker: str = "BTCUSDT", max_orders_per_second: int = 10):
        self.connector = connector
        self.base_ticker = base_ticker
        self.semaphore = asyncio.Semaphore(max_orders_per_second)

    def calculate_order_size(self, target_share: float, current_value: float, total_value: float, spot_price: float) -> float:
        """Расчёт размера ордера в контрактах."""
        target_value = target_share * total_value
        delta_value = target_value - current_value
        order_qty = delta_value / spot_price
        return order_qty

    def round_quantity(self, qty: float, step_size: float) -> float:
        """Математически корректное округление количества до шага лота."""
        if step_size <= 0:
            return qty

        # log10 от 0.001 даст -3. Инвертируем знак для получения количества знаков после запятой.
        precision = max(0, int(round(-math.log10(step_size))))
        return round(qty, precision)

    async def execute_market_order(self, symbol: str, qty: float, side: str, step_size: float = 0.0, reduce_only: bool = False, position_side: str = "BOTH", min_notional: float = 6.0, price: float = 0.0) -> Dict:
        """
        Отправка рыночного ордера на Binance Futures.
        """
        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": f"Округлилось до нуля"}

        # Проверка минимальной стоимости (Notional Value)
        try:
            if price <= 0:
                # Получаем текущую цену только если она не передана
                prices = await self.connector.get_futures_prices([symbol])
                price = prices.get(symbol, 0.0)

            if price > 0 and (qty * price) < min_notional:
                msg = f"Order too small: {qty * price:.2f} USDT < {min_notional} USDT. Skipping."
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
                quantity=abs(qty),
                reduceOnly=reduce_only,
                positionSide=position_side
            )
            logger.info(f"Order: {side} {abs(qty)} {symbol} ({position_side})")
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
        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": f"Округлилось до нуля"}

        try:
            # 1. Получаем mid-price и проверяем стоимость
            order_book = await self.connector.get_order_book(symbol, limit=5)
            best_bid = float(order_book["bids"][0][0])
            best_ask = float(order_book["asks"][0][0])
            mid_price = (best_bid + best_ask) / 2

            if (qty * mid_price) < min_notional:
                msg = f"Limit order too small: {qty * mid_price:.2f} USDT < {min_notional} USDT. Skipping."
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
            logger.info(f"Limit order: {side} {qty} {symbol} @ {limit_price:.6f} (mid={mid_price:.6f}, expected_gain={expected_improvement_pct:.3f}%)")

            # 3. Выставляем POST-ONLY лимитку (гарантия maker-комиссии 0.02%)
            order = await self.connector.place_limit_maker_order(
                symbol=symbol,
                side=side,
                qty=qty,
                price=limit_price,
                position_side=position_side,
                reduce_only=reduce_only
            )

            order_id = order["orderId"]
            logger.info(f"Limit order placed: {order_id}")

            # 4. Ждём исполнения
            filled_qty = 0.0
            avg_fill_price = 0.0

            for _ in range(timeout_sec):
                await asyncio.sleep(1)
                status = await self.connector.get_order_status(symbol, order_id)
                filled_qty = float(status.get("executedQty", 0))
                avg_fill_price = float(status.get("avgPrice", 0)) if status.get("avgPrice") else avg_fill_price

                if status["status"] == "FILLED":
                    # Расчёт выигрыша в цене
                    price_improvement_pct = (avg_fill_price - mid_price) / mid_price * 100 if side == "SELL" else (mid_price - avg_fill_price) / mid_price * 100
                    notional = filled_qty * avg_fill_price
                    profit_usdt = notional * (price_improvement_pct / 100)

                    logger.info(f"Limit FILLED: {filled_qty} @ {avg_fill_price:.6f} | mid={mid_price:.6f} | gain={price_improvement_pct:+.3f}% ({profit_usdt:+.2f} USDT)")
                    return {
                        "status": "SUCCESS_LIMIT",
                        "filled_qty": filled_qty,
                        "avg_price": avg_fill_price,
                        "mid_price": mid_price,
                        "price_improvement_pct": price_improvement_pct,
                        "profit_usdt": profit_usdt,
                        "order_type": "LIMIT_MAKER"
                    }

                # Частичное исполнение — обновляем qty для fallback
                if filled_qty > 0:
                    qty = qty - filled_qty

            # 5. Timeout — отменяем и добиваем market
            logger.warning(f"Limit order timeout ({timeout_sec}s). Cancelling and fallback to market.")
            await self.connector.cancel_order(symbol, order_id)

            # Если было частичное исполнение — логируем
            if filled_qty > 0:
                price_improvement_pct = (avg_fill_price - mid_price) / mid_price * 100 if side == "SELL" else (mid_price - avg_fill_price) / mid_price * 100
                logger.info(f"Partial fill: {filled_qty} @ {avg_fill_price:.6f} | gain={price_improvement_pct:+.3f}%")

            # Добиваем остаток market-ордером
            if qty > 0:
                market_result = await self.execute_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    step_size=0,  # Уже округлено
                    reduce_only=reduce_only,
                    position_side=position_side
                )
                return {
                    "status": "SUCCESS_FALLBACK",
                    "limit_filled_qty": filled_qty,
                    "limit_avg_price": avg_fill_price,
                    "limit_mid_price": mid_price,
                    "limit_price_improvement_pct": price_improvement_pct if filled_qty > 0 else 0,
                    "market_result": market_result,
                    "order_type": "LIMIT_THEN_MARKET"
                }

        except Exception as e:
            logger.error(f"Limit+Fallback FAILED: {e}")
            # Fallback на market в случае ошибки
            try:
                market_result = await self.execute_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    step_size=step_size,
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

        diff_usdt = action["diff_usdt"]
        side = ("BUY" if diff_usdt > 0 else "SELL") if pos_side == "LONG" else ("SELL" if diff_usdt > 0 else "BUY")
        order_qty = abs(diff_usdt / price)
        reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")

        min_notional = portfolio_cfg.get("min_notional_usdt", 6.0) if portfolio_cfg else 6.0

        if limit_order:
            limit_enabled, limit_offset, limit_timeout = self.get_limit_order_params(portfolio_cfg or {})
            res = await self.execute_limit_with_fallback(
                symbol=base_symbol, qty=order_qty, side=side, step_size=step_size,
                reduce_only=reduce_only, position_side=pos_side, offset_pct=limit_offset,
                timeout_sec=limit_timeout, min_notional=min_notional, price=price
            )
        else:
            res = await self.execute_market_order(
                symbol=base_symbol, qty=order_qty, side=side, step_size=step_size,
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

        diff_usdt = action["diff_usdt"]
        side = ("BUY" if diff_usdt > 0 else "SELL") if pos_side == "LONG" else ("SELL" if diff_usdt > 0 else "BUY")
        order_qty = abs(diff_usdt / price)
        reduce_only = (pos_side == "LONG" and side == "SELL") or (pos_side == "SHORT" and side == "BUY")
        step_size = step_sizes.get(base_symbol, 0.0) if step_sizes else 0.0
        min_notional = portfolio_cfg.get("min_notional_usdt", 6.0)

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

            order_value = qty_rounded * price
            commission = order_value * 0.0004

            trade_pnl = 0.0
            if paper_state:
                entry_key = "long_entry_price" if pos_side == "LONG" else "short_entry_price"
                old_entry = paper_state.get(entry_key, price)
                if reduce_only:
                    if pos_side == "LONG":
                        trade_pnl = qty_rounded * (price - old_entry)
                    else:
                        trade_pnl = qty_rounded * (old_entry - price)

            return {
                "type": pos_side,
                "symbol": symbol,
                "side": side,
                "qty": qty_rounded,
                "price": price,
                "status": "SUCCESS",
                "trade_pnl": trade_pnl,
                "commission": commission,
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
                        timeout_sec=limit_timeout, min_notional=min_notional, price=price
                    )
                else:
                    res = await self.execute_market_order(
                        symbol=base_symbol, qty=order_qty, side=side, step_size=step_size,
                        reduce_only=reduce_only, position_side=pos_side, min_notional=min_notional, price=price
                    )

                return {
                    "type": pos_side,
                    "symbol": symbol,
                    "side": side,
                    "qty": order_qty,
                    "price": price,
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
