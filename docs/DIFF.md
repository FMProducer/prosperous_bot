Патч 1/7 — косметика (удалить неиспользуемый импорт)
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
-import os
-import copy
-import json
-import pandas as pd
-import pytest
-import warnings
+import os
+import copy
+import json
+import pandas as pd
+import pytest
 
 # Подавляем известное предупреждение NumPy в сценариях с пустыми срезами
 pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice")
 
 from prosperous_bot.futures_rebalance_backtester import run_backtest, main as backtester_main

Патч 2/7 — Circuit Breaker: «обнуление» портфеля
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
 def test_graceful_handling_of_empty_data(tmp_path):
@@
     assert metrics["status"].lower().startswith("рын")
 
+def test_circuit_breaker_obliteration_returns_status(tmp_path):
+    """
+    Свеча с экстремальным диапазоном и нулевой стартовый NAV ⇒ статус 'Портфель обнулен после АВ'.
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
+    assert "output_dir" in metrics and metrics["output_dir"]

Патч 3/7 — Ветка open == 0 при включённом CB
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
 def test_min_order_notional_skips_dust_orders(tmp_path):
@@
     assert df_tr.empty
+
+def test_open_price_zero_branch_executes(tmp_path):
+    """
+    Ветка: circuit_breaker_threshold_percent > 0 и open == 0 — выполняется без падений и метрики рассчитаны.
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

Патч 4/7 — load_signal_data: пустой/битый/таймзоны + импорт
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
-from prosperous_bot.futures_rebalance_backtester import run_backtest, main as backtester_main
+from prosperous_bot.futures_rebalance_backtester import (
+    run_backtest,
+    main as backtester_main,
+    load_signal_data,
+)
@@
 def test_equity_html_is_generated(tmp_path):
@@
     assert os.path.exists(os.path.join(out_dir, "equity.html"))
 
 def test_metrics_exist_and_reasonable_pf(tmp_path):
@@
     assert "max_drawdown_percent" in metrics
+
+def test_load_signal_data_empty_and_missing_columns(tmp_path):
+    """
+    load_signal_data: (1) пустой CSV → None; (2) без нужных колонок → None.
+    """
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
+    assert str(df2["timestamp"].dt.tz[0]) == "UTC"
+    assert set(df2["signal"].unique()) == {"HOLD", "BUY"}

Патч 5/7 — Safe Mode: вход и выход (гистерезис)
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
 def test_rebalance_trades_csv_empty_when_no_trades(tmp_path):
@@
     assert df_reb.empty
+
+def test_safe_mode_hysteresis_exit(tmp_path):
+    """
+    Проверяет вход в Safe Mode при высоком использовании маржи и выход при снижении (гистерезис).
+    Сценарий: NORMAL 50% USDT / 50% LONG? — нет, задаём 90% LONG, чтобы usage ~45% при левередже 2x.
+    На следующем шаге Safe Mode переводит веса в 100% USDT → на еще одном шаге выходим из Safe Mode.
+    """
+    ts = pd.date_range("2024-08-01", periods=4, freq="h", tz="UTC")
+    df = pd.DataFrame({
+        "timestamp": ts,
+        "open":  [100, 101, 102, 103],
+        "high":  [101, 102, 103, 104],
+        "low":   [ 99, 100, 101, 102],
+        "close": [100, 101, 102, 103],
+        "volume": 1.0
+    })
+    data_csv = tmp_path / "sm.csv"
+    df.to_csv(data_csv, index=False)
+
+    params = {
+        "main_asset_symbol": "BTC",
+        "apply_signal_logic": False,  # чтобы не блокировало сделки
+        "initial_portfolio_value_usdt": 10_000.0,
+        "futures_leverage": 2.0,
+        "commission_taker": 0.0,
+        "slippage_percent": 0.0,
+        "min_order_notional_usdt": 0.0,
+        "min_rebalance_interval_minutes": 0,
+        "rebalance_threshold": 0.0,
+        "target_weights_normal": {"BTC_PERP_LONG": 0.9, "USDT": 0.1},
+        "safe_mode_config": {
+            "enabled": True,
+            "entry_threshold": 0.40,  # usage ~0.45 → войдём
+            "exit_threshold": 0.10,   # после ликвидации позиций usage ~0 → выйдем
+            "target_weights_safe": {"USDT": 1.0}
+        },
+        "circuit_breaker_config": {"threshold_percentage": 10.0},
+        "report_path_prefix": str(tmp_path / "reports"),
+        "use_fixed_report_path": True,
+    }
+    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
+    # Должен быть хотя бы один вход в Safe Mode
+    assert metrics.get("num_safe_mode_entries", 0) >= 1
+    # Safe Mode не должен «залипнуть»: после выхода счетчик времени в SM небольшой (обычно 1 шаг)
+    assert metrics.get("time_steps_in_safe_mode", 0) <= 2
+    # Торги должны были состояться при входе и последующем выходе
+    out_dir = metrics["output_dir"]
+    trades_csv = os.path.join(out_dir, "trades.csv")
+    assert os.path.exists(trades_csv)
+    df_tr = pd.read_csv(trades_csv)
+    assert len(df_tr) >= 2

Патч 6/7 — нейтральный сценарий: нет blocked_trades_log.csv
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
 def test_equity_html_is_generated(tmp_path):
@@
     assert os.path.exists(os.path.join(out_dir, "equity.html"))
 
+def test_no_blocked_trades_log_in_neutral(tmp_path):
+    """
+    При нейтральных условиях (apply_signal_logic=False или HOLD) файл blocked_trades_log.csv не создаётся.
+    """
+    ts = pd.date_range("2024-05-10", periods=4, freq="h", tz="UTC")
+    df = pd.DataFrame({
+        "timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0
+    })
+    data_csv = tmp_path / "neutral.csv"
+    df.to_csv(data_csv, index=False)
+
+    params = {
+        "main_asset_symbol": "BTC",
+        "apply_signal_logic": False,  # нейтральный режим без блокировок
+        "initial_portfolio_value_usdt": 1_000.0,
+        "futures_leverage": 1.0,
+        "commission_taker": 0.0,
+        "slippage_percent": 0.0,
+        "min_order_notional_usdt": 0.0,
+        "min_rebalance_interval_minutes": 0,
+        "rebalance_threshold": 0.0,
+        "target_weights_normal": {"BTC_PERP_LONG": 0.5, "USDT": 0.5},
+        "safe_mode_config": {"enabled": False},
+        "circuit_breaker_config": {"threshold_percentage": 10.0},
+        "report_path_prefix": str(tmp_path / "reports"),
+        "use_fixed_report_path": True,
+    }
+    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
+    out_dir = metrics["output_dir"]
+    blocked_path = os.path.join(out_dir, "blocked_trades_log.csv")
+    assert not os.path.exists(blocked_path)

Патч 7/7 — ранний возврат: date_range фильтрует всё
*** a/tests/test_futures_rebalance_backtester.py
--- b/tests/test_futures_rebalance_backtester.py
@@
 def test_metrics_exist_and_reasonable_pf(tmp_path):
@@
     assert "max_drawdown_percent" in metrics
+
+def test_date_range_filter_returns_empty(tmp_path):
+    """
+    Если после применения date_range данных не остаётся — ранний возврат с нулевыми метриками.
+    """
+    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
+    df = pd.DataFrame({
+        "timestamp": ts,
+        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0
+    })
+    data_csv = tmp_path / "dr.csv"
+    df.to_csv(data_csv, index=False)
+
+    params = {
+        "main_asset_symbol": "BTC",
+        "apply_signal_logic": False,
+        "initial_portfolio_value_usdt": 1000.0,
+        "futures_leverage": 1.0,
+        "commission_taker": 0.0,
+        "slippage_percent": 0.0,
+        "min_order_notional_usdt": 0.0,
+        "min_rebalance_interval_minutes": 0,
+        "rebalance_threshold": 0.0,
+        "target_weights_normal": {"USDT": 1.0},
+        "safe_mode_config": {"enabled": False},
+        "circuit_breaker_config": {"threshold_percentage": 10.0},
+        "date_range": {  # фильтрируем всё вне диапазона
+            "start_date": "2025-01-01T00:00:00Z",
+            "end_date":   "2025-01-01T01:00:00Z",
+        },
+        "report_path_prefix": str(tmp_path / "reports"),
+        "use_fixed_report_path": True,
+    }
+    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)
+    assert metrics["status"] == "Рыночные данные пусты"
+    assert metrics["total_trades"] == 0

Команды для применения по шагам
git checkout -b feature/tests-backtester-rare-branches

# 1/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 1/7>
PATCH
git commit -m "test(backtester): remove unused import in tests file"

# 2/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 2/7>
PATCH
git commit -m "test(backtester): add CB obliteration early-return test"

# 3/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 3/7>
PATCH
git commit -m "test(backtester): cover open==0 branch under Circuit Breaker"

# 4/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 4/7>
PATCH
git commit -m "test(backtester): add load_signal_data tests (empty/missing/tz) and import"

# 5/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 5/7>
PATCH
git commit -m "test(backtester): cover Safe Mode hysteresis (enter & exit)"

# 6/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 6/7>
PATCH
git commit -m "test(backtester): assert no blocked_trades_log.csv in neutral run"

# 7/7
git apply --index <<'PATCH'
<вставьте сюда содержимое Патч 7/7>
PATCH
git commit -m "test(backtester): early-return when date_range filters out all data"

pytest -q --maxfail=1 --disable-warnings \
  --cov=src/prosperous_bot --cov-report=term-missing --cov-fail-under=90