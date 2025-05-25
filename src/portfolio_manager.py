# c:\Python\Prosperous_Bot\src\portfolio_manager.py

from __future__ import annotations
import asyncio, time, logging
from typing import Dict
from exchange_api import ExchangeAPI

_LOG = logging.getLogger(__name__)


class PortfolioManager:
    """
    Снимок портфеля (spot BTC + USDT-фьючерсы) → NAV и notional-веса.
    Кэш снапшота живёт `cache_sec` секунд, чтобы не спамить API.
    """

    def __init__(self, spot: ExchangeAPI, fut: ExchangeAPI, cache_sec: int = 3):
        self.spot, self.fut, self.cache_sec = spot, fut, cache_sec
        self._snap: dict[str, float] | None = None
        self._ts = 0.0

    # ---- внутренний снапшот ----
    async def _snapshot(self) -> dict[str, float]:
        if self._snap and time.time() - self._ts < self.cache_sec:
            return self._snap

        # spot-баланс BTC (sync SDK → to_thread)
        bal = await asyncio.to_thread(self.spot.spot.get_account_detail)
        spot_qty = float(next((a.available for a in bal if a.currency == "BTC"), 0))

        # позиции BTC-USDT futures (sync SDK → to_thread)
        pos = await asyncio.to_thread(self.fut.futures.list_positions, settle="usdt")
        long_q = short_q = upl = 0.0
        for p in pos:
            if p.contract != "BTC_USDT":
                continue
            q = float(p.size)
            upl += float(p.unrealised_pnl)
            if q > 0:
                long_q += q
            elif q < 0:
                short_q += abs(q)

        self._snap = s = dict(spot_qty=spot_qty, long_qty=long_q, short_qty=short_q, upl=upl)
        self._ts = time.time()
        return s

    # ---- NAV ----
    async def nav_usd(self, p_spot: float) -> float:
        s = await self._snapshot()
        return s["spot_qty"] * p_spot + s["upl"]

    # ---- notional-веса ----
    async def get_notional_weights(self, p_spot: float) -> Dict[str, float]:
        s = await self._snapshot()
        nav = await self.nav_usd(p_spot)
        if nav == 0:
            return {k: 0.0 for k in ("BTC_SPOT", "BTC_LONG5X", "BTC_SHORT5X")}
        return {
            "BTC_SPOT": s["spot_qty"] * p_spot / nav,
            "BTC_LONG5X": s["long_qty"] * p_spot / nav,
            "BTC_SHORT5X": s["short_qty"] * p_spot / nav,
        }

    # ---- helper для метрик ----
    async def log_weights(self, p_spot: float) -> None:
        w = await self.get_notional_weights(p_spot)
        _LOG.info("Weights %%: %s", {k: f"{v:.2%}" for k, v in w.items()})