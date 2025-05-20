"""portfolio_manager.py – расчёт NAV и ноциональных весов BTC-нейтрального портфеля."""

from __future__ import annotations
import asyncio, logging, time
from typing import Dict
from adaptive_agent.exchange_api import ExchangeAPI

_LOG = logging.getLogger(__name__)

class PortfolioManager:
    """Служебный слой: берёт снапшот позиций и выдаёт веса BTC-порта по спотовой цене."""

    def __init__(self, spot_api: ExchangeAPI, fut_api: ExchangeAPI, leverage: int = 5):
        self.spot_api, self.fut_api = spot_api, fut_api
        self.leverage = leverage
        self._last_snapshot: dict | None = None
        self._ts_snapshot: float = 0

    # ------------------------------------------------------------------ #
    async def _snapshot(self) -> dict:
        """Возвращает кешированный снимок (spot_qty, long_qty, short_qty, upl)."""
        if self._last_snapshot and time.time() - self._ts_snapshot < 3:
            return self._last_snapshot           # 3-секундный кеш

        # --- spot BTC баланс ---
        bal = await self.spot_api.get_wallet_balance('BTC')
        spot_qty = float(bal.available) if bal else 0.0

        # --- фьючерсные позиции ---
        pos = await self.fut_api.make_request(self.fut_api.futures.list_positions, settle='usdt')
        long_qty = short_qty = upl = 0.0
        for p in pos:
            if p.contract != 'BTC_USDT':           # рассматриваем только BTC
                continue
            qty = float(p.size)
            if qty > 0:
                long_qty += qty
            elif qty < 0:
                short_qty += abs(qty)
            upl += float(p.unrealised_pnl)

        snap = dict(spot_qty=spot_qty, long_qty=long_qty,
                    short_qty=short_qty, upl=upl)
        self._last_snapshot, self._ts_snapshot = snap, time.time()
        return snap

    # ------------------------------------------------------------------ #
    async def _equity_usd(self, p_spot: float) -> float:
        s = await self._snapshot()
        return s["spot_qty"] * p_spot + s["upl"]

    # ------------------------------------------------------------------ #
    async def get_notional_weights(self, p_spot: float) -> Dict[str, float]:
        """
        Возвращает веса ножек портфеля:
        { 'BTC_SPOT': w1, 'BTC_LONG5X': w2, 'BTC_SHORT5X': w3 }
        """
        s = await self._snapshot()
        not_long  = s["long_qty"]  * p_spot
        not_short = s["short_qty"] * p_spot
        nav = await self._equity_usd(p_spot)
        if nav == 0:
            _LOG.warning("NAV == 0; возвращаю нули")
            return {k: 0 for k in ("BTC_SPOT", "BTC_LONG5X", "BTC_SHORT5X")}
        return {
            "BTC_SPOT"   : (s["spot_qty"] * p_spot) / nav,
            "BTC_LONG5X" : not_long  / nav,
            "BTC_SHORT5X": not_short / nav,
        }

    # ------------------------------------------------------------------ #
    async def refresh_and_log(self, p_spot: float):
        """Асинхронно печатает веса – удобно для Prometheus-экспорта."""
        w = await self.get_notional_weights(p_spot)
        _LOG.info("Weights: %s", {k: f"{v:.3%}" for k, v in w.items()})
