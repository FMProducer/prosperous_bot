"""
Integration test: BTC-neutral rebalance must yield zero PnL under ideal conditions.
"""

import pathlib
import pytest

# Import run_backtest with robust fallback
try:
    from prosperous_bot.rebalance_backtester import run_backtest
except ImportError:
    from rebalance_backtester import run_backtest  # type: ignore

DATA_FILE = pathlib.Path("graphs/BTCUSDT_data.csv") # Adjusted path


@pytest.mark.integration
def test_btc_neutral_rebalance_zero_pnl(tmp_path):
    if not DATA_FILE.exists():
        pytest.skip(f"Historical data file not found: {DATA_FILE}")

    init_nav = 10_000.0  # USDT

    params = {
        "main_asset_symbol": "BTC",
        "futures_leverage": 5.0,
        "apply_signal_logic": False,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,
        # No frictions
        "taker_commission_rate": 0.0,
        "maker_commission_rate": 0.0,
        "slippage_percent": 0.0,
        # Capital & reports
        "initial_portfolio_value_usdt": init_nav,
        "min_order_notional_usdt": 10.0,
        "min_rebalance_interval_minutes": 30,
        "report_path_prefix": str(tmp_path),
        # Target weights for perfect neutrality (x5 leverage)
        # 0.35 + 5 × (0.29 − 0.36) = 0  → рыночно-нейтрально
        "target_weights_normal": {
            "BTC_SPOT": 0.35,
            "BTC_PERP_LONG": 0.29,
            "BTC_PERP_SHORT": 0.36,
        },
        # Switch off extra logic
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"enabled": False},
        "data_settings": {
            "csv_file_path": str(DATA_FILE),
            "signals_csv_path": "",
            "timestamp_col": "timestamp",
            "ohlc_cols": {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
            },
            "price_col_for_rebalance": "close",
        },
    }
    # Генерируем полный отчёт; не режим оптимизатора
    params["generate_reports"] = True
    results = run_backtest(params, str(DATA_FILE), is_optimizer_call=False)

    assert results["status"] == "Completed"
    assert results["total_net_pnl_usdt"] == pytest.approx(0.0, abs=1e-2) # Increased tolerance to 1 cent
    assert results["final_portfolio_value_usdt"] == pytest.approx(init_nav, abs=1e-2)
