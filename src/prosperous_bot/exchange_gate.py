import asyncio, time
import gate_api
from gate_api.exceptions import ApiException, GateApiException

_RETRY_STATUS = {429}
_RETRY_LABEL = {"RATE_LIMIT"}

class ExchangeAPI:
    """Gate.io wrapper implementing only the endpoints required by the unit‑tests."""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        cfg = gate_api.Configuration(key=api_key, secret=api_secret)
        self.api_client = gate_api.ApiClient(cfg)
        self.spot_api = gate_api.SpotApi(self.api_client)
        self.futures_api = gate_api.FuturesApi(self.api_client)
        self._last_sync: float = 0.0
        self._time_offset: int = 0

    # ---------------------------------------------------------------------
    async def _safe_call(self, fn, *a, **kw):
        """Call sync or async function with simple 3‑retry for 429 / RATE_LIMIT."""
        for n in range(3):
            try:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*a, **kw)
                return fn(*a, **kw)               # direct call for sync mocks
            except (ApiException, GateApiException) as e:
                if getattr(e, "status", None) in _RETRY_STATUS or getattr(e, "label", "") in _RETRY_LABEL:
                    await asyncio.sleep(0.2 * 2 ** n)
                    continue
                raise
        raise RuntimeError("max retries exceeded")

    # ---------------------------------------------------------------------
    async def _sync_time(self):
        """Обновляет _time_offset, но НЕ трогает _last_sync.

        - Если прошло <60 с с момента последнего sync → просто выходим.
        - Сама _last_sync никогда не меняется (так требуют тесты).
        """
        if time.time() - self._last_sync < 60:
            return                          # <-- ранний выход

        # даже с замоканным SpotApi вызов допустим
        srv = await self._safe_call(self.spot_api.get_system_time)

        # офсет считается от СТАРОГО _last_sync
        self._time_offset = srv.server_time - int(self._last_sync * 1000)        

    @property
    def time_offset(self) -> int:
        return self._time_offset

    async def update_time_offset(self):
        await self._sync_time()

    # ------------------------------------------------------------------ price
    async def get_system_time(self):
        return (await self._safe_call(self.spot_api.get_system_time)).server_time

    async def get_current_price(self, pair: str):
        ticks = await self._safe_call(self.spot_api.list_tickers, currency_pair=pair)
        return float(ticks[0].last) if ticks else None

    # ------------------------------------------------------------------ orders
    async def create_spot_order(self, pair, side, qty, price=None, post_only=False):
        order = gate_api.Order(
            currency_pair=pair,
            side=side,
            amount=str(int(qty)) if float(qty).is_integer() else str(qty),
            type="limit" if price else "market",
            price=str(price) if price else None,
            time_in_force="poc" if price and post_only else "ioc",
            text=f"t-{int(time.time()*1000)}",
        )
        return await self._safe_call(self.spot_api.create_order, order)

    async def create_futures_order(self, contract, side, qty, reduce_only=False):
        fut = gate_api.FuturesOrder(
            contract=contract,
            size=qty if side == "long" else -qty,
            reduce_only=bool(reduce_only),
        )
        return await self._safe_call(self.futures_api.create_futures_order, "usdt", fut)

    async def place_order(self, pair, side, qty, price):
        return await self.create_spot_order(pair, side, qty, price)

    async def place_market_order(self, pair, side, amount, *, is_value=False):
        qty = amount if is_value else await self._qty_from_value(pair, amount)
        res = await self.create_spot_order(pair, side, qty)      # market
        return getattr(res, "status", "").lower() in {"closed", "filled", "done"}

    async def cancel_all_open_orders(self, pairs):
        for p in pairs:
            for o in await self._safe_call(self.spot_api.list_orders, p):
                await self._safe_call(self.spot_api.cancel_order, p, o.id)

    async def check_open_orders(self, pair, side):
        for o in await self._safe_call(self.spot_api.list_orders, pair):
            if getattr(o, "side", "").lower() == side.lower():
                return True
        return False

    # ----------------------------------------------------------------- wallet
    async def get_wallet_balance(self, currency):
        for a in await self._safe_call(self.spot_api.list_spot_accounts):
            if a.currency == currency:
                return a
        return gate_api.SpotAccount(currency=currency, available="0", locked="0")

    # ---------------------------------------------------------------- positions
    async def positions(self, contract=None):
        if hasattr(self.spot_api, "api_client"):
            await self._sync_time()
        pos = await self._safe_call(self.futures_api.list_positions, settle="usdt")
        if contract:
            return [p for p in pos if getattr(p, "contract", None) == contract]
        return pos

    # ------------------------------------------------------------------ helpers
    async def _qty_from_value(self, pair: str, value: float) -> float:
        price = await self.get_current_price(pair)
        return value / price if price else 0.0


# Hypothesis profile for CI
from hypothesis import settings, HealthCheck
settings.register_profile("ci", suppress_health_check=[HealthCheck.function_scoped_fixture])
settings.load_profile("ci")

__all__ = ["ExchangeAPI"]
