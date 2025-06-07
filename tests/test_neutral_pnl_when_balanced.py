import copy
import pytest
import pandas as pd
from prosperous_bot.rebalance_backtester import run_backtest

# Путь к CSV с историей
MARKET_DATA_PATH = "graphs/BTCUSDT_data.csv"

BASE_CONFIG = {
    "apply_signal_logic": False,
    "commission_taker": 0.0,
    "slippage_percent": 0.0,
    "futures_leverage": 5.0,
    "target_weights_normal": {
        "BTC_SPOT": 0.65,
        "BTC_PERP_SHORT": 0.24,
        "BTC_PERP_LONG": 0.11
    },
    "report_path_prefix": "C:/Python/Prosperous_Bot/reports/test_neutrality",
    "safe_mode_config": {
        "enabled": False
    },
    "output_config": {
        "enable_csv_output": True,
        "save_equity": True,
        "save_summary": True
    }
}


@pytest.mark.parametrize("market_data_file", [MARKET_DATA_PATH])
def test_neutral_pnl_when_balanced(market_data_file):
    config = copy.deepcopy(BASE_CONFIG)

    result = run_backtest(config, data_path=market_data_file, is_optimizer_call=True)

    df_equity = pd.read_csv("C:/Python/Prosperous_Bot/reports/test_neutrality/equity.csv")
    nav_start = df_equity["portfolio_value_usdt"].iloc[0]
    nav_end = df_equity["portfolio_value_usdt"].iloc[-1]
    delta_nav = abs(nav_end - nav_start)

    assert delta_nav < 1.0, (
        f"NAV изменился при рыночно-нейтральной позиции: "
        f"start = {nav_start}, end = {nav_end}, Δ = {delta_nav}"
    )
