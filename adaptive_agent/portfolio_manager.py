
import asyncio
from typing import Dict
from gate_api import SpotAccount, Position

class PortfolioManager:
    def __init__(self, spot, fut):
        self.spot = spot
        self.fut = fut
        self._cache = None

    async def _snapshot(self) -> Dict:
        if self._cache:
            return self._cache
        spot_bal = await asyncio.to_thread(self.spot.spot.get_account_detail)
        fut_pos = await asyncio.to_thread(self.fut.futures.list_positions, settle="usdt")
        upl = sum(float(p.unrealised_pnl) for p in fut_pos if p.contract == "BTC_USDT")
        short_qty = sum(int(abs(float(p.size))) for p in fut_pos if float(p.size) < 0 and p.contract == "BTC_USDT")
        long_qty = sum(int(abs(float(p.size))) for p in fut_pos if float(p.size) > 0 and p.contract == "BTC_USDT")
        spot_qty = sum(float(a.available) for a in spot_bal if a.currency == "BTC")
        self._cache = {
            "spot_qty": spot_qty,
            "short_qty": short_qty,
            "long_qty": long_qty,
            "upl": upl,
        }
        return self._cache

    async def get_positions_with_margin(self) -> list:
        pos = await asyncio.to_thread(self.fut.futures.list_positions, settle="usdt")
        return [p for p in pos if p.contract == "BTC_USDT"]

    async def get_value_distribution_usdt(self, p_spot: float, p_contract: float) -> Dict[str, float]:
        s = await self._snapshot()
        return {
            "BTC_SPOT": s["spot_qty"] * p_spot,
            "BTC_SHORT5X": float(int(s["short_qty"])) * p_contract,
            "BTC_LONG5X": float(int(s["long_qty"])) * p_contract
        }
