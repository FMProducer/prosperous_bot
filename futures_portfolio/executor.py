"""Исполнение ордеров на Binance Futures (Hedge Mode)."""
import logging
import asyncio
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PortfolioExecutor:
    def __init__(self, connector, base_ticker: str = "BTCUSDT"):
        self.connector = connector
        self.base_ticker = base_ticker

    def calculate_order_size(self, target_share: float, current_value: float, total_value: float, spot_price: float) -> float:
        """Расчёт размера ордера в контрактах."""
        target_value = target_share * total_value
        delta_value = target_value - current_value
        order_qty = delta_value / spot_price
        return order_qty

    def round_quantity(self, qty: float, step_size: float) -> float:
        """Округление количества до шага лота."""
        if step_size <= 0:
            return qty
        precision = 0
        s = str(step_size).rstrip('0')
        if '.' in s:
            precision = len(s.split('.')[1])
        return round(qty, precision)

    async def execute_market_order(self, symbol: str, qty: float, side: str, step_size: float = 0.0, reduce_only: bool = False, position_side: str = "BOTH", min_notional: float = 6.0) -> Dict:
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
            # Получаем текущую цену для проверки стоимости
            prices = await self.connector.get_futures_prices([symbol])
            price = prices.get(symbol)
            if price and (qty * price) < min_notional:
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
                                           min_notional: float = 6.0) -> Dict:
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
