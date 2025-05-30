import asyncio

class PortfolioManager:
    """Simplified portfolio manager used in unit/property tests."""

    def __init__(self, spot_api, futures_api):
        self.spot_api = spot_api
        self.futures_api = futures_api

    async def get_value_distribution_usdt(self, p_spot: float, p_contract: float | None = None):
        acc_raw = self.spot_api.spot.get_account_detail()
        accounts = await acc_raw if asyncio.iscoroutine(acc_raw) else acc_raw

        spot_qty = sum(float(a.available) for a in accounts if getattr(a, "currency", "") == "BTC")
        spot_val = spot_qty * p_spot

        pos_raw = self.futures_api.futures.list_positions()
        positions = await pos_raw if asyncio.iscoroutine(pos_raw) else pos_raw

        long_val = short_val = 0.0
        for p in positions:
            size = float(p.size)
            margin = float(getattr(p, "margin", 0.0))
            if size > 0:
                long_val += margin
            elif size < 0:
                short_val += margin

        total = spot_val + long_val + short_val or 1.0
        return {
            "BTC_SPOT": spot_val / total,
            "BTC_LONG5X": long_val / total,
            "BTC_SHORT5X": short_val / total,
        }

    def get_value_distribution_sync(self, p_spot: float, p_contract: float):
        return asyncio.run(self.get_value_distribution_usdt(p_spot, p_contract))
