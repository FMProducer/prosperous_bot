import os
import pandas as pd

from prosperous_bot.futures_rebalance_backtester import run_backtest

def test_smoke_run_and_reports(tmp_path):
    # 1) Минимальные рыночные данные (UTC, 12 свечей, 1ч)
    ts = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 100.0 + pd.Series(range(12), dtype=float),
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df.to_csv(data_csv, index=False)

    # 2) Сигналы намеренно в «неотсортированном» порядке,
    #    чтобы проверить сортировку + merge_asof + ffill
    sigs = pd.DataFrame({
        "timestamp": [ts[5], ts[2], ts[8]],
        "signal": ["BUY", "NEUTRAL", "SELL"],
    })
    signals_csv = tmp_path / "signals.csv"
    sigs.to_csv(signals_csv, index=False)

    # 3) Минимальные параметры бэктеста
    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": True,
        "initial_portfolio_value_usdt": 10000.0,
        "futures_leverage": 5.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,  # форсируем стартовый ребаланс
        "target_weights_normal": {
            "BTC_SPOT": 0.34,
            "BTC_PERP_LONG": 0.33,
            "BTC_PERP_SHORT": 0.33,
        },
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 0.0},
        "data_settings": {"signals_csv_path": str(signals_csv)},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,  # отчёты строго в tmp/reports
        "generate_reports_for_optimizer_trial": True,
    }

    # 4) Запуск
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)

    # 5) Базовые проверки результата
    assert isinstance(metrics, dict)
    assert metrics["final_portfolio_value_usdt"] > 0
    for key in ("sharpe_ratio", "profit_factor", "win_rate_percent"):
        assert key in metrics

    # 6) Проверка генерации отчётов
    out_dir = metrics.get("output_dir") or params["report_path_prefix"]
    assert os.path.isdir(out_dir)
    for fn in ("trades.csv", "summary.csv", "equity.csv", "rebalance_trades.csv", "backtest.log"):
        assert os.path.exists(os.path.join(out_dir, fn))
