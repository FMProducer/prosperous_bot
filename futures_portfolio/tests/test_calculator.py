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
    # Share Long = 30000 / (12000 * 5) = 50%
    assert float(calc.share_long_pct) == 50.0
    assert float(calc.share_short_pct) == 0.0
    # Notional Virt = 2000
    # Share Virt = 2000 / 12000 = 16.7%
    assert float(calc.share_virt_pct) == pytest.approx(16.7, abs=0.1)

def test_calculate_deviations(sample_params, targets):
    calc = PortfolioCalculator(**sample_params)
    threshold = 0.05
    actions = calc.calculate_deviations(targets, threshold)
    
    # Shares: L:50%, S:0%, V:16.7%. Targets: L:40%, S:40%, V:20%.
    # Deviations: L: +10% (BREACH), S: -40% (BREACH), V: -3.3% (NO BREACH)
    # Threshold is 5%.
    assert len(actions) == 2
    
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    # Target Notional L = 12000 * 0.4 * 5 = 24000
    # Current Notional L = 30000
    # Diff = 24000 - 30000 = -6000
    assert long_action["diff_usdt"] == pytest.approx(-6000.0)
    
    short_action = next(a for a in actions if a["symbol"] == "BTCUSDT_SHORT")
    # Target Notional S = 12000 * 0.4 * 5 = 24000
    # Current Notional S = 0
    # Diff = 24000 - 0 = 24000
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
    # TPV = 10000 (Equity) + 0.0333 * 66000 (Virtual) = 10000 + 2200 = 12200
    # Wait, real_equity in the test is fixed at 10000. 
    # In reality, real_equity would change with price. 
    # But for unit test of Calculator, we just check its math given the inputs.
    calc = PortfolioCalculator(**params)
    assert float(calc.tpv) == pytest.approx(12200.0)
    # val_long = (0.5 * 60000 / 5) + (0.5 * (66000 - 60000)) = 6000 + 3000 = 9000
    # Share Long = 9000 / 12200 * 100 = 73.8%
    assert float(calc.share_long_pct) == pytest.approx(73.8, abs=0.1)

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
    calc = PortfolioCalculator(**params)
    # tpv = 100000. 
    # notional_long = 10 * 60000 = 600000.
    # target_notional_long = 100000 * 0.4 * 5 = 200000.
    # diff = 200000 - 600000 = -400000.
    actions = calc.calculate_deviations(targets, threshold=0.01, ignore_limits=True)
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    assert long_action["diff_usdt"] == pytest.approx(-400000.0)

def test_limits_deviation(sample_params, targets):
    params = sample_params.copy()
    params["real_equity"] = 100000.0
    params["virt_qty"] = 0.0
    params["positions"] = {"BTCUSDT_LONG": 10.0}
    calc = PortfolioCalculator(**params)
    actions = calc.calculate_deviations(targets, threshold=0.01, ignore_limits=False)
    long_action = next(a for a in actions if a["symbol"] == "BTCUSDT_LONG")
    # TPV is 100,000. max_change = 100,000 * 0.5 * 5 = 250,000.
    assert long_action["diff_usdt"] == pytest.approx(-250000.0)
