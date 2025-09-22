import os
import copy
import json
import pandas as pd
import pytest
pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice")
import warnings

# Подавляем известное предупреждение NumPy в сценариях с пустыми срезами
pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice")

from prosperous_bot.futures_rebalance_backtester import (
    run_backtest,
    main as backtester_main,
)

def test_main_cli_execution(tmp_path, monkeypatch):
    """Проверяет запуск бэктестера из командной строки (без варнингов и с отчётами)."""
    # 1. Создаем базовые данные и конфиг
    data_csv = tmp_path / "data.csv"
    df = pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")], "close": [100]})
    df.to_csv(data_csv, index=False)

    config_json = tmp_path / "config.json"
    params = {
        "backtest_settings": {
            "main_asset_symbol": "BTC",
            "initial_portfolio_value_usdt": 1000.0,
            "futures_leverage": 1.0,
            "commission_taker": 0.0,
            "slippage_percent": 0.0,
            "min_order_notional_usdt": 0.0,
            "min_rebalance_interval_minutes": 0,
            "rebalance_threshold": 0.0,
            "target_weights_normal": {"BTC_PERP_LONG": 1.0},
            "safe_mode_config": {"enabled": False},
            "circuit_breaker_config": {"threshold_percentage": 1.0},
            "data_settings": {"csv_file_path": str(data_csv)},
            "report_path_prefix": str(tmp_path / "cli_reports"),
            "use_fixed_report_path": True,
        }
    }
    with open(config_json, "w") as f:
        json.dump(params, f)

    # 2. Мокаем sys.argv для эмуляции вызова из CLI
    monkeypatch.setattr("sys.argv", ["futures_rebalance_backtester.py", "--config_file", str(config_json)])

    # 3. Запускаем main()
    backtester_main()

    # 4. Проверяем, что отчеты созданы
    report_dir = tmp_path / "cli_reports"
    assert os.path.isdir(report_dir)
    assert os.path.exists(report_dir / "summary.csv")
    assert os.path.exists(report_dir / "equity.html")


def test_graceful_handling_of_empty_data(tmp_path):
    """Проверяет, что при пустых/битых данных возвращается корректный статус без падений."""
    empty_csv = tmp_path / "empty.csv"
    with open(empty_csv, "w") as f:
        f.write("timestamp,close\n")  # Только заголовок

    params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 1000.0,
        "futures_leverage": 1.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,
        "target_weights_normal": {"USDT": 1.0},
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 1.0},
        "report_path_prefix": str(tmp_path),
        "use_fixed_report_path": True,
    }

    # Предупреждения от numpy о пустых срезах подавляем на уровне модуля
    metrics = run_backtest(params, str(empty_csv), is_optimizer_call=False)
    assert isinstance(metrics, dict)
    assert "status" in metrics
    assert metrics["status"].lower().startswith("рын")

def test_circuit_breaker_obliteration_returns_status(tmp_path):
    """
    Свеча с экстремальным диапазоном и нулевой стартовый NAV ⇒
    немедленное завершение с статусом 'Портфель обнулен после АВ'.
    """
    ts = pd.date_range("2024-07-01", periods=1, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0],
        "high": [200.0],   # +100%
        "low":  [  0.0],   # -100%
        "close": [150.0],
        "volume": 1.0
    })
    data_csv = tmp_path / "cb.csv"
    df.to_csv(data_csv, index=False)

    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": False,
        "initial_portfolio_value_usdt": 0.0,        # ключ к ветке 'обнуления'
        "futures_leverage": 5.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 1.0,                 # сделок не будет
        "target_weights_normal": {"USDT": 1.0},
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 0.1},  # 10% порог, свеча >100%
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    assert metrics["status"] == "Портфель обнулен после АВ"
    assert metrics.get("num_circuit_breaker_triggers", 0) >= 1
    assert metrics["max_drawdown_percent"] == -100.0
    # Путь отчёта должен быть определён (пусть и с ранним выходом)
    assert "output_dir" in metrics and metrics["output_dir"]

def test_open_price_zero_branch_executes(tmp_path):
    """
    Ветка: circuit_breaker_threshold_percent > 0 и open == 0.
    Проверяем, что расчёт проходит без падений и метрики отдаются.
    """
    ts = pd.date_range("2024-07-02", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [0.0, 100.0, 101.0],   # первый бар с open=0 → спец-ветка
        "high":  [0.1, 101.0, 102.0],
        "low":   [0.0,  99.0, 100.0],
        "close": [0.05,100.5,101.5],
        "volume": 1.0
    })
    data_csv = tmp_path / "oz.csv"
    df.to_csv(data_csv, index=False)

    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": False,
        "initial_portfolio_value_usdt": 1000.0,
        "futures_leverage": 2.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,
        "target_weights_normal": {"BTC_PERP_LONG": 0.5, "USDT": 0.5},
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 0.1},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    assert isinstance(metrics, dict)
    assert "final_portfolio_value_usdt" in metrics
    assert "sharpe_ratio" in metrics

def test_load_signal_data_empty_and_missing_columns(tmp_path):
    """
    load_signal_data: (1) пустой CSV → None; (2) без нужных колонок → None.
    """
    from prosperous_bot.futures_rebalance_backtester import load_signal_data
    # 1) Пустой файл
    empty_csv = tmp_path / "empty_signals.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)
    assert load_signal_data(str(empty_csv)) is None
    # 2) Нет 'timestamp' или 'signal'
    bad_csv = tmp_path / "bad_signals.csv"
    pd.DataFrame({"time": ["2024-01-01T00:00:00Z"], "sig": ["BUY"]}).to_csv(bad_csv, index=False)
    assert load_signal_data(str(bad_csv)) is None

def test_load_signal_data_timezone_localize_and_convert(tmp_path):
    """
    load_signal_data: (1) наивные timestamps → локализация в UTC;
                     (2) timestamps с TZ → конвертация в UTC; сигнал → upper().
    """
    from prosperous_bot.futures_rebalance_backtester import load_signal_data
    # 1) Наивные метки времени
    ts_naive = ["2024-03-01 00:00:00", "2024-03-01 01:00:00"]
    csv1 = tmp_path / "sig_naive.csv"
    pd.DataFrame({"timestamp": ts_naive, "signal": ["buy", "sell"]}).to_csv(csv1, index=False)
    df1 = load_signal_data(str(csv1))
    assert df1 is not None and not df1.empty
    assert str(df1["timestamp"].dt.tz) == "UTC"
    assert set(df1["signal"].unique()) == {"BUY", "SELL"}
    # 2) Таймштампы с зоной (конвертация в UTC)
    ts_tz = ["2024-03-01T00:00:00+03:00", "2024-03-01T01:00:00+03:00"]
    csv2 = tmp_path / "sig_tz.csv"
    pd.DataFrame({"timestamp": ts_tz, "signal": ["HOLD", "BUY"]}).to_csv(csv2, index=False)
    df2 = load_signal_data(str(csv2))
    assert df2 is not None and not df2.empty
    # обе записи должны быть в UTC (конвертированы)
    assert str(df2["timestamp"].dt.tz) == "UTC"
    assert set(df2["signal"].unique()) == {"HOLD", "BUY"}

def test_blocked_short_close_on_sell_signal(tmp_path):
    """
    SELL-сигнал не должен разрешать закрытие SHORT (покупкой).
    Сценарий: стартуем с SHORT весом, включаем Safe Mode с safe-весами = 100% USDT,
    активные сигналы все 'SELL' → попытка закрытия шорта блокируется и попадает в blocked_trades_log.csv.
    """
    # 1) Рыночные данные — 6 часов, умеренные колебания
    ts = pd.date_range("2024-03-01", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100, 101, 99, 100, 98, 97],
        "high": [101, 102, 100, 101, 99, 98],
        "low":  [ 99, 100, 98,  99, 97, 96],
        "close":[100, 100, 99, 100, 98.5, 97.5],
        "volume": 1.0
    })
    data_csv = tmp_path / "mkt.csv"
    df.to_csv(data_csv, index=False)

    # 2) Сигналы — везде SELL
    df_sig = pd.DataFrame({"timestamp": ts, "signal": "SELL"})
    sig_csv = tmp_path / "sig.csv"
    df_sig.to_csv(sig_csv, index=False)

    # 3) Параметры: стартуем с SHORT 50% / USDT 50%, Safe Mode активируется сразу (порог 0.0),
    # safe-веса переводят портфель в 100% USDT → для закрытия SHORT потребуется BUY, что запрещено SELL-сигналом.
    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": True,
        "initial_portfolio_value_usdt": 10_000.0,
        "futures_leverage": 3.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,
        "target_weights_normal": {"BTC_PERP_SHORT": 0.5, "USDT": 0.5},
        "safe_mode_config": {
            "enabled": True,
            "entry_threshold": 0.0,
            "exit_threshold": 1.0,
            "target_weights_safe": {"USDT": 1.0}
        },
        "circuit_breaker_config": {"threshold_percentage": 1.0},
        "data_settings": {"signals_csv_path": str(sig_csv)},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }

    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    out_dir = metrics["output_dir"]
    blocked_path = os.path.join(out_dir, "blocked_trades_log.csv")
    assert os.path.exists(blocked_path)
    df_blocked = pd.read_csv(blocked_path)
    assert not df_blocked.empty
    # Проверяем, что блокировалась попытка BUY по SHORT на SELL-сигнале
    row = df_blocked.iloc[0]
    assert row["asset_key"] == "BTC_PERP_SHORT"
    assert row["intended_action"] == "BUY"
    assert row["active_signal"] == "SELL"

def test_min_order_notional_skips_dust_orders(tmp_path):
    """
    При слишком малом номинале сделки (ниже min_order_notional_usdt) ордер не исполняется (нет записей в trades.csv).
    """
    ts = pd.date_range("2024-02-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0
    })
    data_csv = tmp_path / "d.csv"
    df.to_csv(data_csv, index=False)

    # Сигналы нейтральные (не нужны)
    sig_csv = tmp_path / "s.csv"
    pd.DataFrame({"timestamp": ts, "signal": "HOLD"}).to_csv(sig_csv, index=False)

    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": True,
        "initial_portfolio_value_usdt": 1_000.0,
        "futures_leverage": 1.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 100.0,  # высокий порог → «пыль»
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,        # форс ребаланс
        "target_weights_normal": {"BTC_PERP_LONG": 0.001, "USDT": 0.999},  # ~1 USDT «пыль»
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 10.0},
        "data_settings": {"signals_csv_path": str(sig_csv)},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    out_dir = metrics["output_dir"]
    trades_csv = os.path.join(out_dir, "trades.csv")
    df_tr = pd.read_csv(trades_csv)
    # Ожидаем отсутствие сделок (все корректировки ниже мин. нотионала)
    assert df_tr.empty

def test_rebalance_trades_csv_empty_when_no_trades(tmp_path):
    """
    Если ни одной сделки не было, rebalance_trades.csv должен существовать, но быть пустым.
    """
    ts = pd.date_range("2024-04-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0
    })
    data_csv = tmp_path / "nt.csv"
    df.to_csv(data_csv, index=False)

    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": False,
        "initial_portfolio_value_usdt": 500.0,
        "futures_leverage": 1.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 1.0,        # очень высокий порог → сделок не будет
        "target_weights_normal": {"USDT": 1.0},
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 10.0},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    out_dir = metrics["output_dir"]
    rebalance_csv = os.path.join(out_dir, "rebalance_trades.csv")
    assert os.path.exists(rebalance_csv)
    df_reb = pd.read_csv(rebalance_csv)
    assert df_reb.empty

def test_equity_html_is_generated(tmp_path):
    """
    Проверяет, что HTML-график equity генерируется для стандартного прогона.
    """
    ts = pd.date_range("2024-05-01", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0 + pd.Series(range(8), dtype=float), "volume": 1.0
    })
    data_csv = tmp_path / "eq.csv"
    df.to_csv(data_csv, index=False)

    sig_csv = tmp_path / "eq_sig.csv"
    pd.DataFrame({"timestamp": ts, "signal": "HOLD"}).to_csv(sig_csv, index=False)

    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": False,
        "initial_portfolio_value_usdt": 2_000.0,
        "futures_leverage": 2.0,
        "commission_taker": 0.000,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,
        "target_weights_normal": {"BTC_PERP_LONG": 0.5, "USDT": 0.5},
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 10.0},
        "data_settings": {"signals_csv_path": str(sig_csv)},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    out_dir = metrics["output_dir"]
    assert os.path.exists(os.path.join(out_dir, "equity.html"))

def test_metrics_exist_and_reasonable_pf(tmp_path):
    """
    Генерируем небольшую серию с колебаниями цены и порогом 0.0,
    чтобы появились сделки и метрики. Проверяем PF>0 и наличие Sharpe/MDD.
    """
    ts = pd.date_range("2024-06-01", periods=6, freq="h", tz="UTC")
    closes = [100, 110, 90, 95, 105, 100]  # и рост, и падение
    df = pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": [c+1 for c in closes], "low": [c-1 for c in closes],
        "close": closes, "volume": 1.0
    })
    data_csv = tmp_path / "pf.csv"
    df.to_csv(data_csv, index=False)

    sig_csv = tmp_path / "pf_sig.csv"
    pd.DataFrame({"timestamp": ts, "signal": "HOLD"}).to_csv(sig_csv, index=False)

    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": False,
        "initial_portfolio_value_usdt": 3_000.0,
        "futures_leverage": 2.0,
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.0,
        "target_weights_normal": {"BTC_PERP_LONG": 0.5, "USDT": 0.5},
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 10.0},
        "data_settings": {"signals_csv_path": str(sig_csv)},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
    # Проверяем наличие метрик и разумность PF
    assert "profit_factor" in metrics and metrics["profit_factor"] is not None
    assert metrics["profit_factor"] >= 0.0
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_percent" in metrics
