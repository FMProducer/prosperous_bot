import asyncio
from decimal import Decimal

class PortfolioManager:
    """Simplified portfolio manager used in unit/property tests."""

    def __init__(self, spot_api, futures_api, base_currency: str = "BTC"):
        self.spot_api = spot_api
        self.futures_api = futures_api
        self._base = base_currency

    async def get_value_distribution_usdt(self, p_spot: Decimal, p_contract: Decimal | None = None, leverage: Decimal = Decimal("5.0")):
        p_spot = Decimal(str(p_spot))
        if p_contract is not None:
            p_contract = Decimal(str(p_contract))
        leverage = Decimal(str(leverage))

        acc_raw = self.spot_api.spot.get_account_detail()
        accounts = await acc_raw if asyncio.iscoroutine(acc_raw) else acc_raw

        spot_qty = sum(Decimal(str(a.available or "0")) for a in accounts if getattr(a, "currency", "") == self._base)
        spot_val = spot_qty * p_spot

        pos_raw = self.futures_api.futures.list_positions()
        positions = await pos_raw if asyncio.iscoroutine(pos_raw) else pos_raw

        long_val = short_val = Decimal("0.0")                      # notional in USDT
        for p in positions:
            size = Decimal(str(p.size))
            margin = Decimal(str(getattr(p, "margin", "0.0")))
            if p_contract is not None:
                notional = abs(size) * p_contract
            else:
                notional = margin * leverage  # fallback if contract price not given
            if size > 0:
                long_val += notional
            elif size < 0:
                short_val += notional

        # Recalculate total based on new notional values for accurate weighting
        total = spot_val + long_val + short_val
        if total == 0:  # Avoid division by zero
            return {f"{self._base}_SPOT": Decimal("0.0"), f"{self._base}_PERP_LONG": Decimal("0.0"), f"{self._base}_PERP_SHORT": Decimal("0.0")}

        return {
            f"{self._base}_SPOT"       : spot_val / total,
            f"{self._base}_PERP_LONG"  : long_val / total,
            f"{self._base}_PERP_SHORT" : short_val / total,
        }

    def get_value_distribution_sync(self, p_spot: Decimal, p_contract: Decimal, leverage: Decimal = Decimal("5.0")):
        return asyncio.run(self.get_value_distribution_usdt(p_spot, p_contract, leverage))
