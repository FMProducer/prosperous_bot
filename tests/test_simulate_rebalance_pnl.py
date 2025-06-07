import pandas as pd
from prosperous_bot.rebalance_backtester import simulate_rebalance

def test_simulate_rebalance_pnl():
    # фиктивные свечи
    data = pd.DataFrame([
        {"close": 100},  # Step 0: Price for opening positions
        {"close": 110},  # Step 1: Price for closing BTC_PERP_LONG
        {"close": 90},   # Step 2: Price for BTC_PERP_SHORT operations
    ])

    orders_by_step = {
        0: [
            {"asset_key": "BTC_PERP_LONG", "side": "buy", "qty": 1},    # Open LONG at 100
            {"asset_key": "BTC_PERP_SHORT", "side": "sell", "qty": 3}   # Open SHORT at 100 (direction: -1)
        ],
        1: [
            {"asset_key": "BTC_PERP_LONG", "side": "sell", "qty": 1}    # Close LONG at 110
                                                                       # PnL = (110 - 100) * 1 * 1 * 5.0 = 50.0
        ],
        2: [
            {"asset_key": "BTC_PERP_SHORT", "side": "buy", "qty": 2},   # Buy to close 2 of 3 SHORT contracts at 90
                                                                       # Original SHORT entry: 100, direction: -1
                                                                       # PnL = (90 - 100) * 2 * (-1) * 5.0 = (-10) * (-10) = 100.0
            {"asset_key": "BTC_PERP_SHORT", "side": "sell", "qty": 1}   # Sell to open 1 new SHORT contract at 90
                                                                       # (This adds to remaining 1 contract from previous, averaging price, or opens new if all closed)
                                                                       # This specific order does not generate PnL itself.
        ]
    }

    trades = simulate_rebalance(data, orders_by_step, leverage=5.0)

    # Expected PnL-generating trades:
    # 1. Closing BTC_PERP_LONG: PnL = 50.0
    # 2. Buying to close part of BTC_PERP_SHORT: PnL = 100.0

    assert len(trades) == 2, f"Expected 2 trades, got {len(trades)}. Trades: {trades}"

    pnl_values = sorted([trade['pnl_gross_quote'] for trade in trades]) # Sort for easier comparison
    expected_pnls = sorted([50.0, 100.0])

    assert pnl_values == expected_pnls, f"Expected PnLs {expected_pnls}, got {pnl_values}"

    for trade in trades:
        assert "pnl_gross_quote" in trade, f"Trade missing 'pnl_gross_quote': {trade}"
        assert trade["pnl_gross_quote"] != 0, f"Trade has zero PnL: {trade}"
        assert "qty" in trade, f"Trade missing 'qty': {trade}"
        assert trade["qty"] > 0, f"Trade has zero or negative qty: {trade}"
        assert "asset_key" in trade
        assert "entry_price" in trade
        assert "exit_price" in trade
        assert "leverage" in trade

    # Check details of the trades
    long_close_trade = next((t for t in trades if t['asset_key'] == 'BTC_PERP_LONG'), None)
    assert long_close_trade is not None
    assert long_close_trade['entry_price'] == 100
    assert long_close_trade['exit_price'] == 110
    assert long_close_trade['qty'] == 1
    assert long_close_trade['pnl_gross_quote'] == 50.0

    # The trade log from simulate_rebalance doesn't include 'side'.
    # We infer the short close trade by its unique PnL or other properties.
    short_close_trade = next((t for t in trades if t['pnl_gross_quote'] == 100.0), None)
    assert short_close_trade is not None
    assert short_close_trade['asset_key'] == 'BTC_PERP_SHORT'
    assert short_close_trade['entry_price'] == 100 # Entry price of the original short position
    assert short_close_trade['exit_price'] == 90  # Exit price for this closing trade
    assert short_close_trade['qty'] == 2          # Quantity closed
    assert short_close_trade['pnl_gross_quote'] == 100.0
