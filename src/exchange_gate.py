# exchange_gate.py
"""
Основной модуль для взаимодействия с API биржи Gate.io (Spot и Futures).
Предоставляет класс ExchangeAPI для аутентифицированного доступа к API,
обработки ограничений скорости запросов, синхронизации времени и выполнения
основных торговых операций.
"""
import asyncio
import time
import logging
import gate_api
from gate_api.exceptions import ApiException, GateApiException

__all__ = ["ExchangeAPI"]

_RETRY_STATUS = {429}
_RETRY_LABEL  = {"RATE_LIMIT"}

class ExchangeAPI:
    """
    Предоставляет интерфейс к API биржи Gate.io (Spot и Futures).

    Обрабатывает аутентификацию API, ограничения скорости через метод _safe_call,
    автоматическую синхронизацию времени с сервером и предоставляет методы
    для получения рыночных данных, создания ордеров и управления позициями.
    """
    def __init__(self, api_key: str, api_secret: str):
        cfg = gate_api.Configuration(key=api_key, secret=api_secret)
        self.api_client  = gate_api.ApiClient(cfg)
        self.spot_api    = gate_api.SpotApi(self.api_client)
        self.futures_api = gate_api.FuturesApi(self.api_client) # Шаг 2
        self._time_offset = 0
        self._last_sync   = 0.0

    # ---------- internal ----------
    async def _safe_call(self, fn, *a, **kw): # Шаг 3
        for i in range(3): # Попытки <= 3
            try:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*a, **kw)
                return await asyncio.to_thread(fn, *a, **kw) # Обертка для синхронных функций
            except (ApiException, GateApiException) as e:
                status = getattr(e, "status", None)
                label  = getattr(e, "label", None)
                if status in _RETRY_STATUS or label in _RETRY_LABEL: # RATE_LIMIT или 429
                    logging.warning(f"Rate limit hit or temporary error ({status=}, {label=}). Retrying (attempt {i+1}/3)...")
                    await asyncio.sleep(0.2 * (2 ** i)) # Экспоненциальный back-off: 0.2s, 0.4s, 0.8s
                    continue
                logging.error(f"API call failed: {e}")
                raise
            except Exception as e: # тест 10
                logging.error(f"An unexpected error occurred during API call: {e}")
                raise
        logging.error("Max retries exceeded for API call.")
        raise RuntimeError("max retries exceeded")

    async def _sync_time(self): # Шаг 4
        current_time = time.time()
        if current_time - self._last_sync < 60: # Sync only if last sync was > 60s ago (тест 12)
            return
        srv = await self._safe_call(self.spot_api.get_system_time)
        self._time_offset = srv.server_time - int(current_time * 1000)
        self._last_sync   = current_time # (тест 9)
        logging.info(f"Time synchronized with server. Offset: {self._time_offset}ms")

    # ---------- public ----------
    async def get_system_time(self): # Шаг 8
        return (await self._safe_call(self.spot_api.get_system_time)).server_time

    async def get_current_price(self, pair: str): # Шаг 8
        ticks = await self._safe_call(self.spot_api.list_tickers, currency_pair=pair)
        return float(ticks[0].last) if ticks and len(ticks) > 0 else None

    async def create_futures_order(self, contract: str, side: str, # Шаг 5
                                   qty: int, reduce_only: bool = False):
        await self._sync_time()
        order = gate_api.FuturesOrder(
            contract=contract,
            size=str(qty if side.lower() == "long" else -qty), # Positive for long, negative for short
            reduce_only=reduce_only,
        )
        return await self._safe_call(self.futures_api.create_futures_order, "usdt", order)

    async def create_spot_order(self, pair: str, side: str, # Шаг 6
                                qty: float, *, price: float | None = None,
                                post_only: bool = False):
        await self._sync_time()
        order = gate_api.Order(
            currency_pair=pair,
            side=side.lower(), # 'buy' or 'sell'
            amount=str(qty),
            type="limit" if price else "market",
            price=str(price) if price else None,
            time_in_force="poc" if price and post_only else "ioc", # post_only for limit, ioc for market/taker limit
        )
        return await self._safe_call(self.spot_api.create_order, order)

    async def positions(self, contract: str | None = None):
    """Возвращает список позиций; если передан *contract* — фильтрует."""
    await self._sync_time()
    pos_list = await self._safe_call(
        self.futures_api.list_positions, settle="usdt"
    )
    if contract:
        def _c(p):       # поддержка и dict, и объект-модели
            return p.get("contract") if isinstance(p, dict) else getattr(p, "contract", None)
        return [p for p in pos_list if _c(p) == contract]
    return pos_list
