import pytest, gate_api, random
from hypothesis import given, settings, strategies as st

from hypothesis import settings, HealthCheck
settings.register_profile(
    "fast", max_examples=200, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("fast")

# используем ту же фикстуру exch из conftest.py
#
# qty: 1‒10_000   side: 'long' | 'short'   reduce_only: bool
qty_st   = st.integers(min_value=1, max_value=10_000)
side_st  = st.sampled_from(["long", "short"])
ro_st    = st.booleans()

@settings(max_examples=200, deadline=None)
@given(qty=qty_st, side=side_st, reduce_only=ro_st)
@pytest.mark.asyncio
async def test_create_futures_order_property(exch, mocker, qty, side, reduce_only):
    """Property-based contract: size sign, reduce_only passthrough."""

    # ---- мок Gate API ----  (sync-функция потому что to_thread)
    def _echo(settle, futures_order):    # ← sync ✅
        return futures_order | {"settle": settle}
    mocker.patch.object(gate_api.FuturesApi,
    "create_futures_order",
    side_effect=_echo)

    res = await exch.create_futures_order("BTC_USDT", side, qty, reduce_only)

    # 1) размер со знаком
    size = int(res["size"])
    assert (side == "long"  and size == +qty) or \
           (side == "short" and size == -qty)

    # 2) reduce_only флаг передаётся без искажения
    assert res["reduce_only"] == reduce_only

    # 3) всегда market-IOC
    assert res["price"] == "0" and res["tif"] == "ioc"