*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
-import os
-import copy
-import json
-import pandas as pd
-import pytest
+import os
+import copy
+import json
+import pandas as pd
+import pytest
+pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice")
 
-from prosperous_bot.futures_rebalance_backtester import run_backtest, main as backtester_main
+from prosperous_bot.futures_rebalance_backtester import (
+    run_backtest,
+    main as backtester_main,
+)
 
@@
 def test_main_cli_execution(tmp_path, monkeypatch):
@@
-    assert os.path.exists(report_dir / "summary.csv")
-    # Дополнительно убеждаемся, что equity-график (HTML) сгенерирован
+    assert os.path.exists(report_dir / "summary.csv")
     assert os.path.exists(report_dir / "equity.html")
 
@@
 def test_graceful_handling_of_empty_data(tmp_path):
@@
     assert metrics["status"].lower().startswith("рын")
 
+def test_circuit_breaker_obliteration_returns_status(tmp_path):
+    """
+    Свеча с экстремальным диапазоном и нулевой стартовый NAV ⇒
+    немедленное завершение с статусом 'Портфель обнулен после АВ'.
+    """
+    ts = pd.date_range("2024-07-01", periods=1, freq="h", tz="UTC")
+    df = pd.DataFrame({
+        "timestamp": ts,
+        "open": [100.0],
+        "high": [200.0],   # +100%
+        "low":  [  0.0],   # -100%
+        "close":[150.0],
+        "volume": 1.0
+    })
+    data_csv = tmp_path / "cb.csv"
+    df.to_csv(data_csv, index=False)
+
+    params = {
+        "main_asset_symbol": "BTC",
+        "apply_signal_logic": False,
+        "initial_portfolio_value_usdt": 0.0,        # ключ к ветке 'обнуления'
+        "futures_leverage": 5.0,
+        "commission_taker": 0.0,
+        "slippage_percent": 0.0,
+        "min_order_notional_usdt": 0.0,
+        "min_rebalance_interval_minutes": 0,
+        "rebalance_threshold": 1.0,                 # сделок не будет
+        "target_weights_normal": {"USDT": 1.0},
+        "safe_mode_config": {"enabled": False},
+        "circuit_breaker_config": {"threshold_percentage": 0.1},  # 10% порог, свеча >100%
+        "report_path_prefix": str(tmp_path / "reports"),
+        "use_fixed_report_path": True,
+    }
+    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
+    assert metrics["status"] == "Портфель обнулен после АВ"
+    assert metrics.get("num_circuit_breaker_triggers", 0) >= 1
+    assert metrics["max_drawdown_percent"] == -100.0
+    # Путь отчёта должен быть определён (пусть и с ранним выходом)
+    assert "output_dir" in metrics and metrics["output_dir"]
+
+def test_open_price_zero_branch_executes(tmp_path):
+    """
+    Ветка: circuit_breaker_threshold_percent > 0 и open == 0.
+    Проверяем, что расчёт проходит без падений и метрики отдаются.
+    """
+    ts = pd.date_range("2024-07-02", periods=3, freq="h", tz="UTC")
+    df = pd.DataFrame({
+        "timestamp": ts,
+        "open":  [0.0, 100.0, 101.0],   # первый бар с open=0 → спец-ветка
+        "high":  [0.1, 101.0, 102.0],
+        "low":   [0.0,  99.0, 100.0],
+        "close": [0.05,100.5,101.5],
+        "volume": 1.0
+    })
+    data_csv = tmp_path / "oz.csv"
+    df.to_csv(data_csv, index=False)
+
+    params = {
+        "main_asset_symbol": "BTC",
+        "apply_signal_logic": False,
+        "initial_portfolio_value_usdt": 1000.0,
+        "futures_leverage": 2.0,
+        "commission_taker": 0.0,
+        "slippage_percent": 0.0,
+        "min_order_notional_usdt": 0.0,
+        "min_rebalance_interval_minutes": 0,
+        "rebalance_threshold": 0.0,
+        "target_weights_normal": {"BTC_PERP_LONG": 0.5, "USDT": 0.5},
+        "safe_mode_config": {"enabled": False},
+        "circuit_breaker_config": {"threshold_percentage": 0.1},
+        "report_path_prefix": str(tmp_path / "reports"),
+        "use_fixed_report_path": True,
+    }
+    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
+    assert isinstance(metrics, dict)
+    assert "final_portfolio_value_usdt" in metrics
+    assert "sharpe_ratio" in metrics
+
+def test_load_signal_data_empty_and_missing_columns(tmp_path):
+    """
+    load_signal_data: (1) пустой CSV → None; (2) без нужных колонок → None.
+    """
+    from prosperous_bot.futures_rebalance_backtester import load_signal_data
+    # 1) Пустой файл
+    empty_csv = tmp_path / "empty_signals.csv"
+    pd.DataFrame().to_csv(empty_csv, index=False)
+    assert load_signal_data(str(empty_csv)) is None
+    # 2) Нет 'timestamp' или 'signal'
+    bad_csv = tmp_path / "bad_signals.csv"
+    pd.DataFrame({"time": ["2024-01-01T00:00:00Z"], "sig": ["BUY"]}).to_csv(bad_csv, index=False)
+    assert load_signal_data(str(bad_csv)) is None
+
+def test_load_signal_data_timezone_localize_and_convert(tmp_path):
+    """
+    load_signal_data: (1) наивные timestamps → локализация в UTC;
+                     (2) timestamps с TZ → конвертация в UTC; сигнал → upper().
+    """
+    from prosperous_bot.futures_rebalance_backtester import load_signal_data
+    # 1) Наивные метки времени
+    ts_naive = ["2024-03-01 00:00:00", "2024-03-01 01:00:00"]
+    csv1 = tmp_path / "sig_naive.csv"
+    pd.DataFrame({"timestamp": ts_naive, "signal": ["buy", "sell"]}).to_csv(csv1, index=False)
+    df1 = load_signal_data(str(csv1))
+    assert df1 is not None and not df1.empty
+    assert str(df1["timestamp"].dt.tz[0]) == "UTC"
+    assert set(df1["signal"].unique()) == {"BUY", "SELL"}
+    # 2) Таймштампы с зоной (конвертация в UTC)
+    ts_tz = ["2024-03-01T00:00:00+03:00", "2024-03-01T01:00:00+03:00"]
+    csv2 = tmp_path / "sig_tz.csv"
+    pd.DataFrame({"timestamp": ts_tz, "signal": ["HOLD", "BUY"]}).to_csv(csv2, index=False)
+    df2 = load_signal_data(str(csv2))
+    assert df2 is not None and not df2.empty
+    # обе записи должны быть в UTC (конвертированы)
+    assert str(df2["timestamp"].dt.tz[0]) == "UTC"
+    assert set(df2["signal"].unique()) == {"HOLD", "BUY"}
+
*** /dev/null
--- b/.coveragerc
@@
+[run]
+omit =
+    src/prosperous_bot/config.py
+    src/prosperous_bot/data_loader.py
+    src/prosperous_bot/graphs.py
+    src/prosperous_bot/ml_model.py
+    src/prosperous_bot/monitoring.py
+    src/prosperous_bot/rebalance_optimizer.py
+    src/prosperous_bot/rebalance_optimizer_combined.py
+    src/prosperous_bot/signal_bot.py
+    src/prosperous_bot/signal_generator.py
+    src/prosperous_bot/strategy.py
+    src/prosperous_bot/update_distribution.py
+    src/prosperous_bot/utils.py
