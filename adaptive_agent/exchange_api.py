# c:\Python\Prosperous_Bot\adaptive_agent\exchange_api.py

"""
Async Gate.io adapter used across the whole Trading-AI stack.
Environment:
    GATE_KEY    – API key with Spot + Futures trading rights
    GATE_SECRET  – API secret
All blocking SDK calls are executed via native *Async* clients
or delegated to a thread-pool (fallback).
"""

from __future__ import annotations
import os, asyncio, logging, time, functools
from typing import Any, Dict, List, Optional

import aiohttp
from gate_api import (
    Configuration, ApiClient,
    SpotApi, FuturesApi,    # ← sync-клиенты
    ApiException as GateApiException,
)

_LOG = logging.getLogger(__name__)
_RETRIES = 3
_TIME_SYNC_SEC = 1800    # 30 мин

class ExchangeAPI:
    """Unified Gate.io adapter (Spot + USDT-Futures, Hedge-mode)."""

    def __init__(self) -> None:
        key = os.getenv("GATE_KEY")
        sec = os.getenv("GATE_SECRET")
        if not (key and sec):
            raise RuntimeError("ENV vars GATE_KEY / GATE_SECRET not set")

        cfg = Configuration(key=key, secret=sec)
        self._client = ApiClient(cfg)
        self.spot = SpotApi(self._client)
        self.futures = FuturesApi(self._client)
        self.session = aiohttp.ClientSession()
        self._time_offset = 0    # мс сервер-клиент
        self._last_sync = 0.0    # unix ts
        _LOG.info("ExchangeAPI initialised (async)")

    # ---- #
    #    Public helper for unit-tests & monitoring    #
    # ---- #
    async def get_system_time(self) -> int:
        """Вернёт серверное время Gate.io в миллисекундах."""
        result = await asyncio.to_thread(self.spot.get_system_time)
        return int(result.server_time)

    # ---- #
    #    Time sync    #
    # ---- #
    async def _sync_time(self) -> None:
        if time.time() - self._last_sync < _TIME_SYNC_SEC:
            return
        try:
            ts = (await asyncio.to_thread(self.spot.get_system_time)).server_time
            self._time_offset = int(ts) - int(time.time() * 1000)
            self._last_sync = time.time()
            _LOG.debug("Time offset %+d ms", self._time_offset)
        except Exception:
            _LOG.exception("Unable to sync time; keeping previous offset")

    # ---- #
    #    Spot methods    #
    # ---- #
    async def get_current_price(self, pair: str) -> float:
        await self._sync_time()
        t = await self._safe_call(self.spot.list_tickers, currency_pair=pair)
        return float(t[0].last)

    async def create_spot_order(
        self,
        pair: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        post_only: bool = True,
    ) -> Dict[str, Any]:
        order = {
            "currency_pair": pair,
            "side": side.lower(),
            "amount": str(qty),
            "type": "limit" if price else "market",
            "price": str(price) if price else "0",
            "time_in_force": "gtc" if price else "ioc",
        }
        if post_only:
            order["iceberg"] = "0"
        return await self._safe_call(self.spot.create_order, order=order)

    # ---- #
    #    Futures methods    #
    # ---- #
    async def positions(self) -> List[Dict[str, Any]]:
        return await self._safe_call(self.futures.list_positions, settle="usdt")

    async def create_futures_order(
        self,
        contract: str,
        side: str,    # "long" / "short"
        qty: int,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        size = str(qty if side == "long" else -qty)
        order = {
            "contract": contract,
            "size": size,
            "price": "0",    # market
            "tif": "ioc",
            "reduce_only": reduce_only,
        }
        return await self._safe_call(
            self.futures.create_futures_order,
            settle="usdt",
            futures_order=order,
        )

    # ---- #
    #    helper / error-handling    #
    # ---- #
    async def _safe_call(self, func, *args, **kwargs):
        for attempt in range(_RETRIES):
            try:
                # выполняем sync-метод в отдельном потоке
                bound = functools.partial(func, *args, **kwargs)
                return await asyncio.to_thread(bound)
            except GateApiException as e:
                if e.status == 429 or "too many" in getattr(e, "message", "").lower():
                    await asyncio.sleep(2 ** attempt)
                    continue
                _LOG.error("Gate API %s – %s", e.status, getattr(e, "message", ""))
                raise
            except Exception:
                _LOG.exception("Unexpected error in Gate call")
                raise
        raise RuntimeError("Retries exceeded for %s", func)

    async def close(self) -> None:
        await self.session.close()
        await self._client.rest_client.pool_manager.close()