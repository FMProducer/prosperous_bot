import asyncio

class PortfolioManager:
    """Simplified portfolio manager used in unit/property tests."""

    def __init__(self, spot_api, futures_api, base_currency: str = "BTC"):
        self.spot_api = spot_api
        self.futures_api = futures_api
        self._base = base_currency

    async def get_value_distribution_usdt(self, p_spot: float, p_contract: float | None = None, leverage: float = 5.0):
        acc_raw = self.spot_api.spot.get_account_detail()
        accounts = await acc_raw if asyncio.iscoroutine(acc_raw) else acc_raw

        spot_qty = sum(float(a.available) for a in accounts if getattr(a, "currency", "") == self._base)
        spot_val = spot_qty * p_spot

        pos_raw = self.futures_api.futures.list_positions()
        positions = await pos_raw if asyncio.iscoroutine(pos_raw) else pos_raw

        long_val = short_val = 0.0                      # notional in USDT
        for p in positions:
            size = float(p.size)
            # notional = abs(size) × contract-price (fallback→margin)
            margin = float(getattr(p, "margin", 0.0))
            notional = abs(margin * leverage)
            if size > 0:
                long_val += notional
            elif size < 0:
                short_val += notional

        # Recalculate total based on new notional values for accurate weighting
        total = spot_val + long_val + short_val
        if total == 0:  # Avoid division by zero
            return {f"{self._base}_SPOT": 0.0, f"{self._base}_PERP_LONG": 0.0, f"{self._base}_PERP_SHORT": 0.0}

        return {
            f"{self._base}_SPOT"       : spot_val / total,
            f"{self._base}_PERP_LONG"  : long_val / total,
            f"{self._base}_PERP_SHORT" : short_val / total,
        }

    def get_value_distribution_sync(self, p_spot: float, p_contract: float, leverage: float = 5.0):
        return asyncio.run(self.get_value_distribution_usdt(p_spot, p_contract, leverage))
