
import logging
from typing import Dict
from adaptive_agent.portfolio_manager import PortfolioManager

TARGET_WEIGHTS = {
    "BTC_SPOT": 0.65,
    "BTC_SHORT5X": 0.24,
    "BTC_LONG5X": 0.11
}

class RebalanceEngine:
    def __init__(self, portfolio: PortfolioManager, threshold_pct: float = 0.01):
        self.portfolio = portfolio
        self.threshold_pct = threshold_pct
        self.logger = logging.getLogger(__name__)

    async def build_orders(self, p_spot: float) -> list[Dict]:
        values = await self.portfolio.get_value_distribution_usdt(p_spot, p_contract=None)
        positions = await self.portfolio.get_positions_with_margin()

        try:
            pos_data = next(p for p in positions if p.contract == "BTC_USDT" and abs(float(p.size)) >= 1)
            p_contract = float(pos_data.margin) / abs(float(pos_data.size))
        except StopIteration:
            raise ValueError("No valid BTC_USDT position with margin and size ≥ 1")

        total_value = sum(values.values())
        orders = []

        for asset, target_weight in TARGET_WEIGHTS.items():
            current_value = values.get(asset, 0.0)
            actual_weight = current_value / total_value
            delta_pct = abs(actual_weight - target_weight)

            if delta_pct < self.threshold_pct:
                continue

            delta_value = (target_weight - actual_weight) * total_value
            side = "BUY" if delta_value > 0 else "SELL"

            if asset == "BTC_SPOT":
                qty = abs(delta_value) / p_spot
                symbol = "BTC_USDT"
                px = p_spot * (0.999 if side == "BUY" else 1.001)
                orders.append(dict(
                    symbol=symbol,
                    side=side,
                    qty=round(qty, 6),
                    px=px,
                    post_only=True
                ))
            else:
                qty_contracts = int(round(abs(delta_value) / p_contract))
                if qty_contracts < 1:
                    qty_contracts = 1
                symbol = {
                    "BTC_SHORT5X": "BTC5S_USDT",
                    "BTC_LONG5X": "BTC5L_USDT"
                }[asset]
                px = p_contract * (0.999 if side == "BUY" else 1.001)
                orders.append(dict(
                    symbol=symbol,
                    side=side,
                    qty=qty_contracts,
                    px=px,
                    post_only=True
                ))

        self.logger.info("Build orders: %s", orders)
        return orders
