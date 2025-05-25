
TARGET_WEIGHTS = {
    "BTC_SPOT": 0.65,
    "BTC_SHORT5X": 0.24,
    "BTC_LONG5X": 0.11,
}

class RebalanceEngine:
    """Формирует ордера для приведения распределения к целевым долям."""
    def __init__(self, portfolio, *, threshold_pct: float = 0.02):
        self.portfolio = portfolio
        self.threshold_pct = threshold_pct

    async def build_orders(self, *, p_spot: float, p_contract: float | None = None):
        dist = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract)
        orders = []
        for key, target in TARGET_WEIGHTS.items():
            current = dist.get(key, 0.0)
            diff = target - current
            if abs(diff) > self.threshold_pct:
                qty = max(1, int(abs(diff) * 100))
                symbol = "BTC_USDT" if key != "BTC_SPOT" else "BTC"
                side = "buy" if diff > 0 else "sell"
                orders.append({"symbol": symbol, "side": side, "qty": qty})
        return orders
