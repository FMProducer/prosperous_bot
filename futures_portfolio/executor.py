"""Исполнение ордеров на Binance Futures (Hedge Mode)."""
import logging
from typing import Dict

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

    async def execute_market_order(self, symbol: str, qty: float, side: str, step_size: float = 0.0, reduce_only: bool = False, position_side: str = "BOTH") -> Dict:
        """
        Отправка рыночного ордера на Binance Futures.
        qty - количество контрактов
        side - "BUY" или "SELL"
        position_side - "LONG" или "SHORT" (для Hedge Mode)
        """
        if qty == 0:
            return {"status": "NO_ORDER", "message": "Размер ордера равен нулю"}

        if step_size > 0:
            qty = self.round_quantity(qty, step_size)
            if qty == 0:
                return {"status": "NO_ORDER", "message": f"Округлилось до нуля"}

        try:
            import asyncio
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
