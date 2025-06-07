import pandas as pd
from prosperous_bot.rebalance_backtester import simulate_rebalance

def test_simulate_rebalance_pnl():
    # фиктивные свечи
    data = pd.DataFrame([
        {"close": 100},
        {"close": 110},  # close long
        {"close": 90},   # close short
    ])

    orders_by_step = {
        0: [{"asset_key": "BTC_PERP_LONG", "side": "buy", "qty": 1}],
        1: [{"asset_key": "BTC_PERP_LONG", "side": "sell", "qty": 1}],
        2: [{"asset_key": "BTC_PERP_SHORT", "side": "buy", "qty": 2}],
        2: [{"asset_key": "BTC_PERP_SHORT", "side": "sell", "qty": 2}]
    }

    trades = simulate_rebalance(data, orders_by_step, leverage=5.0)
    assert len(trades) >= 1
    for trade in trades:
        assert "pnl_gross_quote" in trade
        assert trade["pnl_gross_quote"] != 0
