import pytest
from decimal import Decimal
from futures_portfolio.calculator import PortfolioCalculator

@pytest.fixture
def targets():
    return {
        "BASE_LONG": {"share": 0.4, "leverage": 5.0},
        "BASE_SHORT": {"share": 0.4, "leverage": 5.0},
        "VIRTUAL": {"share": 0.2, "leverage": 1.0}
    }

@pytest.fixture
def sample_params(targets):
    return {
        "positions": {"BTCUSDT_LONG": 0.5, "BTCUSDT_SHORT": 0.0},
        "spot_price": 60000.0,
        "real_equity": 10000.0,
        "virt_qty": 0.03333333333333333, # Equivalent to 2000 USDT at 60k
        "long_entry_price": 60000.0,
        "short_entry_price": 0.0,
        "base_ticker": "BTCUSDT",
        "siphoning_reserve": 0.0,
        "targets": targets
    }

def test_calculator_initialization(sample_params):
    calc = PortfolioCalculator(**sample_params)
    # TPV = 10000 (Equity) + 2000 (Virtual) = 12000
    assert float(calc.tpv) == pytest.approx(12000.0)
    assert float(calc.total_tpv) == pytest.approx(12000.0)
    
    # Notional Long = 0.5 * 60000 = 30000
    # Val Long = Margin (30000/5=6000) + PnL (0) = 6000
    # Share Long = 6000 / 12000 = 50%
    assert float(calc.share_long_pct) == 50.0
    assert float(calc.share_short_pct) == 0.0
    # Notional Virt = 2000
    # Share Virt = 2000 / 12000 = 16.67%
    assert float(calc.share_virt_pct) == pytest.approx(16.67, abs=0.1)

def test_calculate_deviations(sample_params, targets):
    calc = PortfolioCalculator(**sample_params)
    threshold = 0.05
    res = calc.calculate_deviations(targets, threshold_surplus=threshold, threshold_deficit=threshold, current_equity=calc.tpv)
    actions = res["actions"]
    
    # Shares: L:50%, S:0%, V:16.7%. Targets: L:40%, S:40%, V:20%.
    # Deviations: L: +10% (BREACH), S: -40% (BREACH), V: -3.3% (NO BREACH)
    # Threshold is 5%.
    assert len(actions) == 2
    
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    # Current Share L = 50%. Target Share L = 40%. Diff Share = 10%.
    # Diff USDT = -0.1 * 12000 * 5 = -6000.
    assert long_action["diff_usdt"] == pytest.approx(-6000.0)
    
    short_action = next(a for a in actions if a["symbol"] == "BTCUSDT_SHORT")
    # TPV = 12000. Target share S = 40%. Current share S = 0%.
    # diff_share = -0.4.
    # diff_usdt = -(-0.4) * 12000 * 5 = 24000.
    # BUT: val_cash = 12000 - 6000 (long margin) - 2000 (virt) = 4000.
    # available_funds starts at 4000.
    # Needed equity for short = 0.4 * 12000 = 4800.
    # available_funds = 4000 (cash) + 1200 (proceeds from long reduction) = 5200.
    # Since 5200 >= 4800, we get full size: 0.4 * 12000 * 5 = 24000.
    assert short_action["diff_usdt"] == pytest.approx(24000.0)
    
    # Virtual should not be here since 3.3% < 5%
    assert not any(a["symbol"] == "VIRTUAL" for a in actions)

def test_siphoning_reserve_impact(sample_params):
    params = sample_params.copy()
    params["siphoning_reserve"] = 1000.0
    calc = PortfolioCalculator(**params)
    assert float(calc.total_tpv) == pytest.approx(13000.0)
    assert float(calc.tpv) == pytest.approx(12000.0)

def test_price_change_impact(sample_params):
    params = sample_params.copy()
    params["spot_price"] = 66000.0
    # l_qty = 0.5, long_entry_price = 60000.0. mtm_pnl_l = 0.5 * (66000 - 60000) = 3000.
    # real_equity = 10000.
    # virt_qty = 0.03333..., virt_value = 0.0333... * 66000 = 2200.
    # TPV = 10000 + 3000 + 2200 = 15200.
    calc = PortfolioCalculator(**params)
    assert float(calc.tpv) == pytest.approx(15200.0)
    # val_long = (0.5 * 60000 / 5) + 3000 = 9000.
    # Share Long = 9000 / 15200 * 100 = 59.21%
    assert float(calc.share_long_pct) == pytest.approx(59.21, abs=0.1)

def test_negative_tpv_protection():
    calc = PortfolioCalculator(
        positions={},
        spot_price=60000.0,
        real_equity=-1000.0,
        virt_qty=0.01 # 600 USDT
    )
    # tpv = -1000 + 600 = -400 -> protected to 1e-9
    assert float(calc.tpv) == pytest.approx(1e-9)

def test_ignore_limits_deviation(sample_params, targets):
    params = sample_params.copy()
    params["real_equity"] = 100000.0
    params["virt_qty"] = 0.0
    params["positions"] = {"BTCUSDT_LONG": 10.0}
    # val_long = (10 * 60000 / 5) = 120000.
    # tpv = 100000 (real_equity) + 0 (mtm_pnl) + 0 (virt) = 100000.
    # share_long = 120000 / 100000 = 1.2
    # target_share_long = 0.4
    # diff_share = 1.2 - 0.4 = 0.8
    # diff_usdt = -0.8 * 100000 * 5 = -400000.
    calc = PortfolioCalculator(**params)
    res = calc.calculate_deviations(targets, threshold_surplus=0.01, threshold_deficit=0.01, current_equity=calc.tpv, ignore_limits=True)
    actions = res["actions"]
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    assert long_action["diff_usdt"] == pytest.approx(-400000.0)

def test_limits_deviation(sample_params, targets):
    params = sample_params.copy()
    params["real_equity"] = 100000.0
    params["virt_qty"] = 0.0
    params["positions"] = {"BTCUSDT_LONG": 10.0}
    calc = PortfolioCalculator(**params)
    res = calc.calculate_rebalance(targets, threshold_surplus=0.01, threshold_deficit=0.01, current_equity=calc.tpv, ignore_limits=False)
    actions = res["actions"]
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    # TPV is 100,000. diff_usdt is -400,000 as calculated above.
    assert long_action["diff_usdt"] == pytest.approx(-400000.0)

def test_anti_churn_fuse(targets):
    # last_reb = 60000. Target share LONG = 40%.
    # Current share LONG = 50%. (surplus)
    # Price must be >= last_reb * (1 + threshold) to allow sell.
    params = {
        "positions": {"BTCUSDT_LONG": 0.5, "BTCUSDT_SHORT": 0.0},
        "spot_price": 60300.0, # 0.5% increase
        "last_rebalance_price": 60000.0,
        "real_equity": 10000.0,
        "virt_qty": 0.0,
        "long_entry_price": 60000.0,
        "targets": targets,
        "min_notional": 6.0
    }
    # TPV = 10000 + 0.5*(60300-60000) = 10150.
    # val_long = (0.5*60000/5) + 150 = 6150.
    # share_long = 6150 / 10150 = 60.59%
    # threshold_surplus = 0.01 (1%)
    # limit_price = 60000 * 1.01 = 60600.
    # spot_price 60300 < 60600 -> Surplus action should be BLOCKED.
    calc = PortfolioCalculator(**params)
    res = calc.calculate_rebalance(targets, threshold_surplus=0.01, threshold_deficit=0.01, current_equity=calc.tpv)
    assert not any(a["symbol"] == "BTCUSDT_LONG" for a in res["actions"])
    
    # Now price at 60700 (> 60600)
    params["spot_price"] = 60700.0
    calc = PortfolioCalculator(**params)
    res = calc.calculate_rebalance(targets, threshold_surplus=0.01, threshold_deficit=0.01, current_equity=calc.tpv)
    assert any(a["symbol"] == "BTCUSDT_LONG" for a in res["actions"])

def test_pnl_protection_mode(targets):
    # allow_surplus_sell=False
    params = {
        "positions": {"BTCUSDT_LONG": 0.5, "BTCUSDT_SHORT": 0.0},
        "spot_price": 60000.0,
        "real_equity": 5000.0, # Low equity, total PnL < 0
        "initial_capital": 10000.0,
        "virt_qty": 0.0,
        "long_entry_price": 60000.0,
        "targets": targets,
        "min_notional": 6.0
    }
    calc = PortfolioCalculator(**params)
    # share_long = 6000 / 5000 = 120%. target=40%. surplus=80%.
    res = calc.calculate_rebalance(targets, threshold_surplus=0.01, threshold_deficit=0.01, current_equity=calc.tpv, allow_surplus_sell=False)
    assert not any(a["is_reduction"] for a in res["actions"])

def test_force_block_net_move_guard(targets):
    params = {
        "positions": {"BTCUSDT_LONG": 0.5, "BTCUSDT_SHORT": 0.0},
        "spot_price": 60000.0,
        "real_equity": 10000.0,
        "virt_qty": 0.0,
        "targets": targets
    }
    calc = PortfolioCalculator(**params)
    res = calc.calculate_rebalance(targets, threshold_surplus=0.01, threshold_deficit=0.01, current_equity=calc.tpv, force_block=True)
    assert len(res["actions"]) == 0

def test_edge_cases(targets):
    # entry_price=0
    calc = PortfolioCalculator(
        positions={"BTCUSDT_LONG": 1.0},
        spot_price=50000.0,
        real_equity=10000.0,
        virt_qty=0.0,
        long_entry_price=0.0,
        targets=targets
    )
    # val_long = ((l_qty * Decimal(str(long_entry_price)) / l_lev) + mtm_pnl_l)
    # If long_entry_price is 0, (1.0 * 0 / 5) + (1.0 * (50000 - 0)) = 50000.
    assert float(calc.val_long) == 50000.0
    
    # initial TPV=0
    calc = PortfolioCalculator(positions={}, spot_price=50000.0, real_equity=0.0, virt_qty=0.0, initial_capital=0.0)
    assert float(calc.tpv) == 1e-9
    
    # shares sum to 100%
    assert float(calc.share_long_pct + calc.share_short_pct + calc.share_virt_pct + calc.share_cash_pct) == 100.0

def test_virt_pnl_paths(targets):
    # virt_entry_price > 0
    calc = PortfolioCalculator(
        positions={}, spot_price=50000.0, real_equity=10000.0, virt_qty=0.1,
        virt_entry_price=40000.0, targets=targets
    )
    assert float(calc.pnl_v) == pytest.approx(1000.0) # (50000-40000)*0.1
    
    # virt_debt > 0
    calc = PortfolioCalculator(
        positions={}, spot_price=50000.0, real_equity=10000.0, virt_qty=0.1,
        virt_entry_price=0.0, virt_debt=4500.0, targets=targets
    )
    assert float(calc.pnl_v) == pytest.approx(500.0) # (50000*0.1)-4500
