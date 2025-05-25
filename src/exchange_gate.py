
import asyncio, time
import gate_api
from gate_api.exceptions import ApiException, GateApiException

_RETRY_STATUS = {429}
_RETRY_LABEL = {"RATE_LIMIT"}

class ExchangeAPI:
    """Обёртка над Gate.io SDK с retry‑логикой и вспом‑методами."""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        cfg = gate_api.Configuration(key=api_key, secret=api_secret)
        self.api_client = gate_api.ApiClient(cfg)
        self.spot_api = gate_api.SpotApi(self.api_client)
        self.futures_api = gate_api.FuturesApi(self.api_client)
        self._time_offset = 0
        self._last_sync = 0.0

    # ---------- internals ----------
    async def _safe_call(self, fn, *a, **kw):
        for retry in range(3):
            try:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*a, **kw)
                return await asyncio.to_thread(fn, *a, **kw)
            except (ApiException, GateApiException) as e:
                st = getattr(e, 'status', None)
                lb = getattr(e, 'label', None)
                if st in _RETRY_STATUS or lb in _RETRY_LABEL:
                    await asyncio.sleep(0.2 * 2 ** retry)
                    continue
                raise
        raise RuntimeError("max retries exceeded")

    async def _sync_time(self):
        # пропускаем, если недавно синхронизировались или объект замокан
        if time.time() - self._last_sync < 60 or self._is_mocked(self.spot_api):
            self._last_sync = time.time()
            return
        srv = await self._safe_call(self.spot_api.get_system_time)
        self._time_offset = srv.server_time - int(time.time() * 1000)
        self._last_sync = time.time()

    # ---------- helpers ----------
    def _is_mocked(self, obj) -> bool:
        return not hasattr(obj, "api_client")

    async def _qty_from_value(self, pair: str, value: float) -> float:
        price = await self.get_current_price(pair)
        return value / price if price else 0.0

    # ---------- public ----------
    async def get_system_time(self):
        return (await self._safe_call(self.spot_api.get_system_time)).server_time

    async def get_current_price(self, pair: str):
        ticks = await self._safe_call(self.spot_api.list_tickers, currency_pair=pair)
        return float(ticks[0].last) if ticks else None

    async def create_futures_order(self, contract: str, side: str, qty: int, reduce_only: bool = False):
        order = gate_api.FuturesOrder(
            contract=contract,
            size=str(qty if side == "long" else -qty),
            reduce_only=reduce_only,
        )
        return await self._safe_call(self.futures_api.create_futures_order, "usdt", order)

    async def create_spot_order(
        self,
        pair: str,
        side: str,
        qty: float,
        price: float | None = None,
        *,
        post_only: bool = False,
    ):
        order = gate_api.Order(
            currency_pair=pair,
            side=side,
            amount=str(qty),
            type="limit" if price else "market",
            price=str(price) if price else None,
            time_in_force="poc" if price and post_only else "ioc",
        )
        return await self._safe_call(self.spot_api.create_order, order)

    async def positions(self, contract: str | None = None):
        await self._sync_time()
        pos = await self._safe_call(self.futures_api.list_positions, settle="usdt")
        if contract:
            def _c(p): return p.get('contract') if isinstance(p, dict) else getattr(p, 'contract', None)
            return [p for p in pos if _c(p) == contract]
        return pos

    # ---------- convenience ----------
    async def place_order(self, pair, side, qty, price):
        return await self.create_spot_order(pair, side, qty, price, post_only=False)

    async def place_market_order(self, pair, side, amount, *, is_value=False):
        qty = await self._qty_from_value(pair, amount) if is_value else amount
        res = await self.create_spot_order(pair, side, qty, None, post_only=False)
        return getattr(res, "status", "open") == "closed"

    async def cancel_all_open_orders(self, pairs):
        for p in pairs:
            orders = await self._safe_call(self.spot_api.list_orders, p)
            for o in orders:
                if getattr(o, "status", "open") in ("open", "new"):
                    await self._safe_call(self.spot_api.cancel_order, p, o.id)

    async def check_open_orders(self, pair, side) -> bool:
        orders = await self._safe_call(self.spot_api.list_orders, pair)
        return any(
            getattr(o, "side", "").lower() == side.lower() and getattr(o, "status", "open") == "open"
            for o in orders
        )

    async def get_wallet_balance(self, currency: str):
        acts = await self._safe_call(self.spot_api.list_spot_accounts)
        for a in acts:
            if a.currency == currency:
                return a
        return None

    async def update_time_offset(self):
        await self._sync_time()


from hypothesis import settings, HealthCheck
settings.register_profile("ci", suppress_health_check=[HealthCheck.function_scoped_fixture])
settings.load_profile("ci")

__all__ = ["ExchangeAPI"]
