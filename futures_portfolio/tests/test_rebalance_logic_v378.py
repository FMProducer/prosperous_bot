
import pytest
from decimal import Decimal
from calculator import PortfolioCalculator

def test_rebalance_logic_v378():
    # Targets based on User's model
    targets = {
        "BASE_LONG": {"share": 0.29, "leverage": 5},
        "BASE_SHORT": {"share": 0.36, "leverage": 5},
        "VIRTUAL": {"share": 0.35, "leverage": 1}
    }
    initial_cap = 100.0
    threshold = 0.035 # 3.5%
    
    print("\n--- CASE 1: Perfect Balance ---")
    # Equity: L=29, S=36, V=35, C=0. TPV=100.
    # Prices match entry to have 0 PnL
    # Qty * Price / Lev = 145*1/5 = 29
    calc = PortfolioCalculator(
        positions={"BTCUSDT_LONG": 145.0, "BTCUSDT_SHORT": -180.0},
        spot_price=1.0,
        real_equity=65.0, # 29 (L margin) + 36 (S margin) + 0 (Cash)
        virt_qty=35.0,
        long_entry_price=1.0,
        short_entry_price=1.0,
        base_ticker="BTCUSDT",
        initial_capital=initial_cap,
        targets=targets
    )
    res = calc.calculate_rebalance(targets, threshold, threshold, current_equity=calc.tpv)
    print(f"DEBUG: L:{res['share_long_pct']}% S:{res['share_short_pct']}% V:{res['share_virt_pct']}% C:{res['share_cash_pct']}%")
    if res["actions"]:
        print(f"DEBUG Actions: {res['actions']}")
    assert len(res["actions"]) == 0
    assert res["share_long_pct"] == 29.0
    assert res["share_short_pct"] == 36.0
    assert res["share_virt_pct"] == 35.0
    assert res["share_cash_pct"] == 0.0
    assert res["tpv"] == 100.0

    print("--- CASE 2: LONG Surplus (Sell to Cash) ---")
    # Price rises to 1.1 (+10%)
    # Long Equity: 29 + (145 * 0.1) = 29 + 14.5 = 43.5
    # Short Equity: 36 - (180 * 0.1) = 36 - 18 = 18.0
    # Virtual Equity: 35 * 1.1 = 38.5
    # Cash: 0
    # TPV: 43.5 + 18.0 + 38.5 + 0 = 100.0 (Market Neutral!)
    # Shares: L:43.5%, S:18.0%, V:38.5%
    # Long exceeds threshold: abs(43.5 - 29.0) = 14.5% > 3.5%
    # Short exceeds threshold: abs(18.0 - 36.0) = 18.0% > 3.5%
    # Virtual exceeds threshold: abs(38.5 - 35.0) = 3.5% >= 3.5% (Triggered)
    
    calc = PortfolioCalculator(
        positions={"BTCUSDT_LONG": 145.0, "BTCUSDT_SHORT": -180.0},
        spot_price=1.2,
        real_equity=65.0, # (29 cost) + (36 cost)
        virt_qty=35.0,
        long_entry_price=1.0,
        short_entry_price=1.0,
        base_ticker="BTCUSDT",
        initial_capital=initial_cap,
        targets=targets,
        last_rebalance_price=1.0
    )
    # TPV Calculation: 65 (bal) + (145 * 0.2 PnL_L) + (180 * -0.2 PnL_S) + (35 * 1.2 Virt)
    # TPV = 65 + 29 - 36 + 42 = 100. (Neutral!)
    # V Share = 42 / 100 = 42%. Threshold 3.5%. Target 35%. 42-35 = 7% > 3.5%. SHOULD TRIGGER.
    
    res = calc.calculate_rebalance(targets, threshold, threshold, current_equity=calc.tpv)
    print(f"DEBUG Case 2: L:{res['share_long_pct']}% S:{res['share_short_pct']}% V:{res['share_virt_pct']}% C:{res['share_cash_pct']}%")
    actions = res["actions"]
    if actions:
        print(f"DEBUG Actions Case 2: {actions}")
    
    # Check priorities: L and V should be Priority 0 (SELL), S should be Priority 2 (BUY)
    assert any(a["key"] == "BASE_LONG" and a["priority"] == 0 for a in actions)
    assert any(a["key"] == "VIRTUAL" and a["priority"] == 0 for a in actions)
    assert any(a["key"] == "BASE_SHORT" and a["priority"] == 2 for a in actions)
    
    # Sort order check (SELLs first)
    assert actions[0]["priority"] == 0
    assert actions[-1]["priority"] == 2

    print("--- CASE 3: Cash Guard (Insufficient funds for Short) ---")
    calc = PortfolioCalculator(
        positions={"BTCUSDT_LONG": 145.0, "BTCUSDT_SHORT": -160.0}, # Short is smaller
        spot_price=1.0,
        real_equity=62.0, # 29 + 32 + 1
        virt_qty=34.0,
        long_entry_price=1.0,
        short_entry_price=1.0,
        base_ticker="BTCUSDT",
        initial_capital=initial_cap,
        targets=targets,
        last_rebalance_price=1.0
    )
    calc.val_cash = Decimal('1.0') # Force low cash
    calc.tpv = calc.val_long + calc.val_short + calc.val_virt + calc.val_cash
    
    res = calc.calculate_rebalance(targets, threshold, threshold, current_equity=calc.tpv)
    assert len(res["actions"]) == 0 

    print("--- CASE 4: Priority BUY (Virtual gets cash first) ---")
    calc = PortfolioCalculator(
        positions={"BTCUSDT_LONG": 145.0, "BTCUSDT_SHORT": -100.0}, # Short deficit: 100*1/5 = 20 (Target 36)
        spot_price=1.0,
        real_equity=50.0, # (29 L cost) + (20 S cost) + (1 Cash)
        virt_qty=20.0, # V deficit: 20 (Target 35)
        long_entry_price=1.0,
        short_entry_price=1.0,
        base_ticker="BTCUSDT",
        initial_capital=initial_cap,
        targets=targets,
        last_rebalance_price=1.0
    )
    # TPV = 50 (bal) + 0 (PnL) + 20 (V) = 70.
    # Needs to be TPV=100 for easy math.
    calc.tpv = Decimal('100.0')
    calc.val_long = Decimal('29.0')
    calc.val_short = Decimal('20.0') # Deficit 16
    calc.val_virt = Decimal('20.0')  # Deficit 15
    calc.val_cash = Decimal('31.0') # Total sum = 100
    
    # Force deviations > 3.5%
    calc.share_long_raw = Decimal('0.29')
    calc.share_short_raw = Decimal('0.20')
    calc.share_virt_raw = Decimal('0.20')
    
    # Available Cash is 7. V needs 15, S needs 16.
    calc.val_cash = Decimal('7.0')
    res = calc.calculate_rebalance(targets, threshold, threshold, current_equity=calc.tpv)
    actions = res["actions"]
    print(f"DEBUG Case 4 Shares: L:{res['share_long_pct']}% S:{res['share_short_pct']}% V:{res['share_virt_pct']}% C:{res['share_cash_pct']}%")
    print(f"DEBUG Case 4 Actions: {actions}")
    
    # V should get 7.0 (partial, but first). S should get 0 (no money left).
    v_act = next(a for a in actions if a["key"] == "VIRTUAL")
    assert v_act["diff_usdt"] == 7.0
    assert not any(a["key"] == "BASE_SHORT" for a in actions)
    
    print("--- CASE 5: Heartbeat Summation ---")
    calc = PortfolioCalculator(
        positions={"BTCUSDT_LONG": 100.0, "BTCUSDT_SHORT": -50.0},
        spot_price=1.2,
        real_equity=50.0,
        virt_qty=10.0,
        long_entry_price=1.0,
        short_entry_price=1.0,
        base_ticker="BTCUSDT",
        initial_capital=100.0,
        targets=targets
    )
    res = calc.calculate_rebalance(targets, threshold, threshold, current_equity=calc.tpv)
    sum_vals = res["val_long"] + res["val_short"] + res["val_virt"] + res["val_cash"]
    assert abs(sum_vals - res["tpv"]) < 0.001
    
    print("\n✅ ALL TESTS PASSED!")

if __name__ == "__main__":
    test_rebalance_logic_v378()
