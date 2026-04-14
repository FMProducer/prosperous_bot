"""Исполнение ордеров на Binance Futures."""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PortfolioExecutor:
    def __init__(self, connector):
        self.connector = connector

    def calculate_order_size(self, target_share: float, current_value: float, total_value: float, spot_price: float) -> float:
        """
        Расчёт размера ордера в контрактах.
        target_share - целевая доля (например, 0.4)
        current_value - текущая стоимость позиции в USDT
        total_value - общая стоимость портфеля в USDT
        spot_price - текущая спот-цена символа
        """
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

    async def execute_market_order(self, symbol: str, qty: float, side: str, step_size: float = 0.0, reduce_only: bool = False) -> Dict:
        """
        Отправка рыночного ордера на Binance Futures.
        qty - количество контрактов (может быть дробным)
        side - "BUY" или "SELL"
        step_size - минимальный шаг лота для этого символа
        reduce_only - если True, ордер может только уменьшить позицию
        """
        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        # Применяем округление
        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": f"Размер ордера {qty} после округления до {step_size} равен нулю"}

        try:
            # Binance Futures API: futures_create_order
            # Using asyncio.to_thread for synchronous call
            import asyncio
            result = await asyncio.to_thread(
                self.connector.futures_client.futures_create_order,
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=abs(qty),  # Binance требует положительное количество
                reduceOnly=reduce_only,
            )
            logger.info(f"Order executed: {side} {abs(qty)} {symbol} (rounded to step {step_size}, reduceOnly={reduce_only})")
            return {"status": "SUCCESS", "result": result}
        except Exception as e:
            logger.error(f"Error in execute_market_order: {e}")
            return {"status": "ERROR", "message": str(e)}