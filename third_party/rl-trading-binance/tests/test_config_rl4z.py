import json
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def config_path() -> Path:
    # Try to find the config relative to the test file
    project_root = Path(__file__).resolve().parent.parent
    config = project_root / "user_data" / "config_rl4z.json"
    if not config.exists():
        # Fallback: Try CWD (useful if directory structure is complex)
        config = Path.cwd() / "user_data" / "config_rl4z.json"
    return config

@pytest.fixture(scope="session")
def rl4z_config(config_path: Path):
    assert config_path.exists(), f"Config file not found: {config_path}"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def test_rl_thresholds_are_integers_and_valid(rl4z_config):
    cfg = rl4z_config
    assert isinstance(cfg.get("rl_long_threshold"), int)
    assert isinstance(cfg.get("rl_short_threshold"), int)
    assert cfg["rl_long_threshold"] >= 1
    assert cfg["rl_short_threshold"] >= 1

def test_rl_ensemble_structure_and_values(rl4z_config):
    """Comprehensive check for the 'rl_ensemble' section."""
    ens = rl4z_config.get("rl_ensemble", {})
    assert ens, "rl_ensemble section is missing"

    # Epsilon thresholds
    eps_l = ens.get("epsilon_threshold_long")
    eps_s = ens.get("epsilon_threshold_short")
    assert isinstance(eps_l, (float, int)), "epsilon_threshold_long must be a number"
    assert isinstance(eps_s, (float, int)), "epsilon_threshold_short must be a number"
    assert 0.0 < eps_l < 1.0, "epsilon_threshold_long must be between 0 and 1"
    assert 0.0 < eps_s < 1.0, "epsilon_threshold_short must be between 0 and 1"

    # Dynamic Epsilon aggression
    dd_k = ens.get("dd_aggression_k")
    assert isinstance(dd_k, (float, int)), "dd_aggression_k must be a number"
    assert dd_k > 0, "dd_aggression_k must be positive"

    # Regime filter
    assert isinstance(ens.get("use_regime_filter"), bool), "use_regime_filter must be a boolean"

    # Q-value auto-tuning interval
    assert isinstance(ens.get("q_update_interval"), int), "q_update_interval must be an integer"
    assert ens.get("q_update_interval") >= 0, "q_update_interval cannot be negative"

    # Q-normalization
    qn = ens.get("q_normalization", {})
    assert qn, "q_normalization section is missing"
    for name in ["long_1", "long_2", "short_1", "short_2"]:
        assert name in qn, f"Missing q_normalization for {name}"
        entry = qn[name]
        assert "q_min" in entry and "q_max" in entry
        assert entry["q_max"] >= entry["q_min"]

def test_dynamic_slots_config_sane(rl4z_config):
    ds = rl4z_config.get("dynamic_slots", {})
    assert "enabled" in ds
    assert ds.get("min_slots_per_side", 0) >= 0
    assert ds.get("update_interval_sec", 0) >= 0
    assert ds.get("aggression_factor", 0) > 0

def test_min_quote_volume_and_whitelist(rl4z_config):
    cfg = rl4z_config
    assert cfg.get("min_quote_volume_usd", 0) > 0
    ex = cfg.get("exchange", {})
    wl = ex.get("pair_whitelist", [])
    assert isinstance(wl, list)
    assert len(wl) > 0

def test_general_trading_settings(rl4z_config):
    cfg = rl4z_config
    assert cfg.get("stake_currency") == "USDT"
    assert cfg.get("trading_mode") == "futures"
    assert cfg.get("margin_mode") == "isolated"
    assert isinstance(cfg.get("stake_amount"), (int, float)) or cfg.get("stake_amount") == "unlimited"
    assert isinstance(cfg.get("max_open_trades"), int)

def test_leverage_structure(rl4z_config):
    lev = rl4z_config.get("leverage", {})
    assert isinstance(lev, dict)
    assert "*" in lev
    assert isinstance(lev["*"], int)

def test_order_types_configuration(rl4z_config):
    ot = rl4z_config.get("order_types", {})
    required_keys = ["entry", "exit", "stoploss", "stoploss_on_exchange"]
    for k in required_keys:
        assert k in ot, f"Missing order_type key: {k}"
    assert ot.get("stoploss_price_type") in ["mark", "last", "index"]

def test_rl_feature_flags(rl4z_config):
    cfg = rl4z_config
    flags = [
        "deep_inference", "rl_calibration_mode", "rl_enable_veto",
        "rl_enable_long_1", "rl_enable_long_2",
        "rl_enable_short_1", "rl_enable_short_2"
    ]
    for flag in flags:
        assert flag in cfg, f"Missing RL flag: {flag}"
        assert isinstance(cfg[flag], bool)

def test_pricing_config(rl4z_config):
    for section in ["entry_pricing", "exit_pricing"]:
        pricing = rl4z_config.get(section, {})
        assert "price_side" in pricing
        assert "use_order_book" in pricing
        assert isinstance(pricing["order_book_top"], int)

def test_system_settings(rl4z_config):
    """Checks for system-level settings like CPU threads."""
    threads = rl4z_config.get("cpu_threads")
    assert isinstance(threads, int)
    assert threads > 0
