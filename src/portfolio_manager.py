


async def get_value_distribution_usdt(self, *, p_spot: float, p_contract: float):
    bal = await self.spot_api.list_spot_accounts()
    pos = await self.futures_api.list_positions(settle="usdt")
    spot_qty = next((float(a.available) for a in bal if a.currency == "BTC"), 0.0)
    spot_val = spot_qty * p_spot
    long_val = sum(abs(float(p.size)) * p_contract for p in pos if float(p.size) > 0)
    short_val = sum(abs(float(p.size)) * p_contract for p in pos if float(p.size) < 0)
    total = spot_val + long_val + short_val or 1
    return {
        "BTC_SPOT": spot_val / total,
        "BTC_LONG5X": long_val / total,
        "BTC_SHORT5X": short_val / total,
    }
