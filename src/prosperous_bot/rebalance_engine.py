
class RebalanceEngine:
    """Формирует ордера для приведения распределения к целевым долям."""
    def __init__(self, portfolio, target_weights: dict, spot_asset_symbol: str, futures_contract_symbol_base: str, *, threshold_pct: float = 0.02):
        self.portfolio = portfolio
        self.target_weights = target_weights
        self.spot_asset_symbol = spot_asset_symbol
        self.futures_contract_symbol_base = futures_contract_symbol_base
        self.threshold_pct = threshold_pct

    async def build_orders(self, *, p_spot: float, p_contract: float | None = None):
        dist = await self.portfolio.get_value_distribution_usdt(p_spot=p_spot, p_contract=p_contract)
        orders = []
        for asset_key, target_weight in self.target_weights.items():
            current_weight = dist.get(asset_key, 0.0)
            diff = target_weight - current_weight
            if abs(diff) > self.threshold_pct:
                qty_to_trade = max(1, int(abs(diff) * 100))
                if asset_key.endswith("_SPOT"):
                    order_symbol = self.spot_asset_symbol
                else:
                    order_symbol = self.futures_contract_symbol_base
                side = "buy" if diff > 0 else "sell"
                orders.append({"symbol": order_symbol, "side": side, "qty": qty_to_trade})
        return orders
