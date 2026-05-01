import pytest
from futures_portfolio.calculator import PortfolioCalculator

@pytest.fixture
def targets():
    return {
        "BASE_LONG": {"share": 0.4, "leverage": 5.0},
        "BASE_SHORT": {"share": 0.4, "leverage": 5.0},
        "VIRTUAL": {"share": 0.2}
    }

@pytest.fixture
def sample_params(targets):
    return {
        "positions": {"BTCUSDT_LONG": 0.5, "BTCUSDT_SHORT": 0.0},
        "spot_price": 60000.0,
        "real_equity": 10000.0,
        "virt_basis_price": 60000.0,
        "virt_allocated_usdt": 2000.0,
        "long_entry_price": 60000.0,
        "short_entry_price": 0.0,
        "base_ticker": "BTCUSDT",
        "siphoning_reserve": 0.0,
        "targets": targets
    }

def test_calculator_initialization(sample_params):
    calc = PortfolioCalculator(**sample_params)
    assert calc.tpv == 10000.0
    assert calc.total_tpv == 10000.0
    # New logic: (0.5 * 60000 / 5) / 10000 = 6000 / 10000 = 60%
    assert calc.share_long_pct == 60.0
    assert calc.share_short_pct == 0.0
    assert calc.share_virt_pct == 20.0

def test_calculate_deviations(sample_params, targets):
    calc = PortfolioCalculator(**sample_params)
    threshold = 0.05
    actions = calc.calculate_deviations(targets, threshold)
    # Now it should return all 3 legs if any exceeded
    assert len(actions) == 3
    assert any(a["symbol"] == "BTCUSDT_LONG" for a in actions)
    assert any(a["symbol"] == "BTCUSDT_SHORT" for a in actions)
    assert any(a["symbol"] == "VIRTUAL" for a in actions)

    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    # Current share 0.6, target 0.4. diff = 0.2.
    # diff_usdt = -0.2 * 10000 * 5 = -10000.
    assert long_action["diff_usdt"] == pytest.approx(-10000.0)
    short_action = next(a for a in actions if a["symbol"] == "BTCUSDT_SHORT")
    # Current share 0.0, target 0.4. diff = -0.4.
    # diff_usdt = -(-0.4) * 10000 * 5 = 20000.
    assert short_action["diff_usdt"] == pytest.approx(20000.0)

def test_siphoning_reserve_impact(sample_params):
    params = sample_params.copy()
    params["siphoning_reserve"] = 1000.0
    # initial_capital is 10000 by default. real_equity is 10000.
    # total_tpv = 10000 (equity) + 0 (virt_pnl) + 1000 (safe) = 11000.
    # total_tpv > initial_capital, so tpv = 10000.
    calc = PortfolioCalculator(**params)
    assert calc.total_tpv == 11000.0
    assert calc.tpv == 10000.0
    # notional = 0.5 * 60000 = 30000. val_long = 30000 / 5 = 6000.
    # share_long = 6000 / 10000 = 60%.
    assert calc.share_long_pct == 60.0

def test_price_change_impact(sample_params):
    params = sample_params.copy()
    params["spot_price"] = 66000.0
    params["initial_capital"] = 20000.0
    calc = PortfolioCalculator(**params)
    # total_tpv = 10000 + (2000 * (66/60) - 2000) + 0 = 10000 + 200 = 10200.
    # tpv = 10200 (since < 20000).
    assert calc.total_tpv == 10200.0
    assert calc.tpv == 10200.0
    # notional = 0.5 * 66000 = 33000. val_long = 33000 / 5 = 6600.
    # share_long = 6600 / 10200 = 64.705... -> 64.7%
    assert calc.share_long_pct == 64.7
    # val_virt = 2000 * 1.1 = 2200.
    # share_virt = 2200 / 10200 = 21.568... -> 21.6%
    assert calc.share_virt_pct == 21.6

def test_short_pnl_logic(sample_params):
    params = sample_params.copy()
    params["positions"] = {"BTCUSDT_SHORT": -1.0}
    params["short_entry_price"] = 60000.0
    params["spot_price"] = 54000.0
    # virt_pnl = 2000 * (54/60 - 1) = 2000 * -0.1 = -200.
    # real_equity = 10000.
    # Wait, real_equity should includes unrealized PnL from shorts in real mode?
    # In this test real_equity is fixed at 10000.
    # total_tpv = 10000 - 200 = 9800.
    calc = PortfolioCalculator(**params)
    assert calc.total_tpv == 9800.0
    # notional_short = 1.0 * 54000 = 54000.
    # val_short = 54000 / 5 = 10800.
    # share_short = 10800 / 9800 = 110.2%
    assert calc.share_short_pct == round(10800 / 9800 * 100, 1)

def test_negative_tpv_protection():
    calc = PortfolioCalculator(
        positions={},
        spot_price=60000.0,
        real_equity=-1000.0,
        virt_basis_price=60000.0,
        virt_allocated_usdt=2000.0
    )
    assert calc.tpv == 1e-9

def test_ignore_limits_deviation(sample_params, targets):
    params = sample_params.copy()
    params["real_equity"] = 100000.0
    params["virt_allocated_usdt"] = 20000.0
    params["positions"] = {"BTCUSDT_LONG": 10.0}
    params["initial_capital"] = 150000.0
    calc = PortfolioCalculator(**params)
    # total_tpv = 100000 + 0 + 0 = 100000.
    # tpv = 100000.
    # notional_long = 10 * 60000 = 600000.
    # val_long = 600000 / 5 = 120000.
    # current_share = 120000 / 100000 = 1.2.
    # target_share = 0.4. diff = 0.8.
    # diff_usdt = -0.8 * 100000 * 5 = -400000.
    actions = calc.calculate_deviations(targets, threshold=0.01, ignore_limits=True)
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    assert long_action["diff_usdt"] == pytest.approx(-400000.0)

def test_limits_deviation(sample_params, targets):
    params = sample_params.copy()
    params["real_equity"] = 100000.0
    params["virt_allocated_usdt"] = 20000.0
    params["positions"] = {"BTCUSDT_LONG": 10.0}
    params["initial_capital"] = 150000.0
    calc = PortfolioCalculator(**params)
    actions = calc.calculate_deviations(targets, threshold=0.01, ignore_limits=False)
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    # TPV is 100,000. max_change = 100,000 * 0.5 * 5 = 250,000.
    assert long_action["diff_usdt"] == pytest.approx(-250000.0)
