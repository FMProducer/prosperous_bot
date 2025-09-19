import os
import copy
import json
import pandas as pd
import pytest

from prosperous_bot.futures_rebalance_backtester import run_backtest, main as backtester_main

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

def test_signal_logic_adherence(tmp_path):
    """Проверяет, что логика сигналов (apply_signal_logic=True) соблюдается."""
    # 1. Сценарий: Сигнал 'BUY' должен блокировать продажу LONG-позиции

    # 1.1. Данные: цена растет, чтобы вес BTC_PERP_LONG превысил таргет и вызвал продажу
    ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df_market = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": [100, 100, 130, 130, 130],  # Цена растет на 3-й свече
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df_market.to_csv(data_csv, index=False)

    # 1.2. Сигнал: 'BUY' подается в момент, когда ребалансировка захочет продать
    df_signals = pd.DataFrame({
        "timestamp": [ts[2]],
        "signal": ["BUY"],
    })
    signals_csv = tmp_path / "signals.csv"
    df_signals.to_csv(signals_csv, index=False)

    # 1.3. Параметры: включаем логику сигналов и порог ребаланса
    params = {
        "main_asset_symbol": "BTC",
        "apply_signal_logic": True,
        "initial_portfolio_value_usdt": 10000.0,
        "futures_leverage": 1.0, # Плечо 1 для простоты
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
        "min_order_notional_usdt": 0.0,
        "min_rebalance_interval_minutes": 0,
        "rebalance_threshold": 0.05,  # 5% порог
        "target_weights_normal": {
            "BTC_PERP_LONG": 0.5,
            "USDT": 0.5,
        },
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"threshold_percentage": 0.0},
        "data_settings": {"signals_csv_path": str(signals_csv)},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
    }

    # 1.4. Запуск
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)

    # 1.5. Проверка
    out_dir = metrics["output_dir"]
    
    # Убедимся, что лог заблокированных сделок создан и не пуст
    blocked_log_path = os.path.join(out_dir, "blocked_trades_log.csv")
    assert os.path.exists(blocked_log_path)
    df_blocked = pd.read_csv(blocked_log_path)
    assert not df_blocked.empty

    # Проверяем, что была заблокирована именно продажа LONG
    blocked_trade = df_blocked.iloc[0]
    assert blocked_trade["asset_key"] == "BTC_PERP_LONG"
    assert blocked_trade["intended_action"] == "SELL"
    assert blocked_trade["active_signal"] == "BUY"

    # Убедимся, что в основном логе сделок не было продаж BTC_PERP_LONG
    trades_log_path = os.path.join(out_dir, "trades.csv")
    df_trades = pd.read_csv(trades_log_path)
    sell_long_trades = df_trades[
        (df_trades["asset_type"] == "BTC_PERP_LONG") & (df_trades["action"] == "SELL")
    ]
    assert sell_long_trades.empty

def test_circuit_breaker_trigger(tmp_path):
    """Проверяет срабатывание Circuit Breaker при высокой волатильности."""
    # 1. Данные: одна свеча с аномальным движением (high-low)/open > 30%
    ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df_market = pd.DataFrame({
        "timestamp": ts,
        "open":  [100, 100, 100, 100, 100],
        "high":  [101, 101, 140, 101, 101], # Аномальный скачок high
        "low":   [99,  99,  60,  99,  99],  # Аномальный скачок low
        "close": [100, 100, 100, 100, 100], # Цена закрытия стабильна
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df_market.to_csv(data_csv, index=False)

    # 2. Параметры: включаем Circuit Breaker с порогом 20%
    params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 10000.0,
        "rebalance_threshold": 0.01, # Низкий порог, чтобы ребаланс точно требовался
        "target_weights_normal": {"BTC_SPOT": 1.0}, # Простая стратегия для чистоты теста
        "circuit_breaker_config": {
            "enabled": True, 
            "threshold_percentage": 0.20 # 20% порог
        },
        "data_settings": {},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
        "apply_signal_logic": False, # Отключаем логику сигналов
        "safe_mode_config": {"enabled": False}, # Отключаем Safe Mode
    }

    # 3. Запуск
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)

    # 4. Проверка
    # Убедимся, что счетчик срабатываний CB равен 1
    assert metrics["num_circuit_breaker_triggers"] == 1

    # Проверяем, что на аномальной свече (ts[2]) не было сделок
    # Первая сделка - это начальная закупка на ts[0]
    out_dir = metrics["output_dir"]
    trades_log_path = os.path.join(out_dir, "trades.csv")
    df_trades = pd.read_csv(trades_log_path)
    
    # Должна быть только одна сделка - первоначальная закупка
    assert len(df_trades) == 1
    initial_trade_ts = pd.to_datetime(df_trades.iloc[0]["timestamp_open"])
    assert initial_trade_ts == ts[0]

def test_safe_mode_activation_and_deactivation(tmp_path):
    """Проверяет вход и выход из Safe Mode по порогу использования маржи."""
    # 1. Данные: цена падает, чтобы вызвать рост margin usage, затем восстанавливается
    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df_market = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": [100, 90, 80, 70, 60, 70, 80, 90, 100, 100], # Падение и восстановление
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df_market.to_csv(data_csv, index=False)

    # 2. Параметры: включаем Safe Mode, НЕЙТРАЛЬНАЯ стратегия, высокий порог ребаланса
    params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 10000.0,
        "futures_leverage": 5.0,
        "rebalance_threshold": 1.0, # Отключаем ребаланс после входа
        "target_weights_normal": {
            "BTC_PERP_LONG": 0.8, # Не-нейтральная стратегия для генерации PnL
            "USDT": 0.2,
        },
        "safe_mode_config": {
            "enabled": True,
            "entry_threshold": 0.25, # Порог входа 25%
            "exit_threshold": 0.20,  # Порог выхода 20%
            "target_weights_safe": {
                "BTC_PERP_LONG": 0.1, # Снижаем риски
                "USDT": 0.9
            }
        },
        "data_settings": {},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
        "apply_signal_logic": False,
        "circuit_breaker_config": {"enabled": False},
    }

    # 3. Запуск
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)

    # 4. Проверка
    # В этом сценарии система будет входить в Safe Mode дважды
    assert metrics["num_safe_mode_entries"] == 2
    assert metrics["time_steps_in_safe_mode"] > 0

    # Проверяем, что были сделки, соответствующие переходам
    out_dir = metrics["output_dir"]
    trades_log_path = os.path.join(out_dir, "trades.csv")
    df_trades = pd.read_csv(trades_log_path)

    # Ожидаем 5 сделок: 
    # 1. Начальная закупка
    # 2. Вход в Safe Mode (SELL)
    # 3. Выход из Safe Mode (BUY)
    # 4. Повторный вход в Safe Mode (SELL)
    # 5. Повторный выход из Safe Mode (BUY)
    assert len(df_trades) == 5

def test_rebalance_threshold_and_interval(tmp_path):
    """Проверяет работу порога ребалансировки и минимального интервала."""
    # 1. Данные: цена сначала меняется мало, потом сильно, потом снова сильно
    ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df_market = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": [100, 101, 130, 150, 100], # 101->130(>5%), 130->150(>5%)
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df_market.to_csv(data_csv, index=False)

    # 2. Параметры: порог 5%, интервал 120 минут (2 свечи)
    params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 10000.0,
        "futures_leverage": 1.0,
        "rebalance_threshold": 0.05, # 5% порог
        "min_rebalance_interval_minutes": 120,
        "target_weights_normal": {
            "BTC_PERP_LONG": 0.5,
            "USDT": 0.5,
        },
        "data_settings": {},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
        "apply_signal_logic": False,
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"enabled": False},
    }

    # 3. Запуск
    metrics = run_backtest(params, str(data_csv), is_optimizer_call=False)

    # 4. Проверка
    out_dir = metrics["output_dir"]
    trades_log_path = os.path.join(out_dir, "trades.csv")
    df_trades = pd.read_csv(trades_log_path)

    # Ожидаем ровно 3 сделки:
    # 1. Начальная ребалансировка в ts[0]
    # 2. Ребалансировка в ts[2], когда порог превышен
    # 3. Ребалансировка в ts[4], когда порог снова превышен и интервал прошел
    # В ts[1] порог не превышен, а в ts[3] должен сработать интервал
    assert len(df_trades) == 3

    trade_timestamps = pd.to_datetime(df_trades["timestamp_open"]).tolist()
    assert ts[0] in trade_timestamps
    assert ts[2] in trade_timestamps
    assert ts[4] in trade_timestamps

def test_fees_and_slippage_impact_on_pnl(tmp_path):
    """Проверяет, что комиссии и проскальзывание корректно вычитаются из PnL."""
    # 1. Данные и базовые параметры, вызывающие несколько сделок
    ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df_market = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": [100, 110, 100, 110, 100], # Волатильность для генерации сделок
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df_market.to_csv(data_csv, index=False)

    base_params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 10000.0,
        "rebalance_threshold": 0.05,
        "target_weights_normal": {"BTC_SPOT": 0.5, "USDT": 0.5},
        "data_settings": {},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
        "apply_signal_logic": False,
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"enabled": False},
        "futures_leverage": 1.0,
    }

    # 2. Прогон БЕЗ комиссий
    params_no_fees = copy.deepcopy(base_params)
    params_no_fees["commission_taker"] = 0.0
    params_no_fees["slippage_percent"] = 0.0
    params_no_fees["report_path_prefix"] = str(tmp_path / "reports_no_fees")
    metrics_no_fees = run_backtest(params_no_fees, str(data_csv), is_optimizer_call=False)

    # 3. Прогон С комиссиями
    params_with_fees = copy.deepcopy(base_params)
    params_with_fees["commission_taker"] = 0.001 # 0.1%
    params_with_fees["slippage_percent"] = 0.0005 # 0.05%
    params_with_fees["report_path_prefix"] = str(tmp_path / "reports_with_fees")
    metrics_with_fees = run_backtest(params_with_fees, str(data_csv), is_optimizer_call=False)

    # 4. Проверка
    # Итоговый PnL с комиссиями должен быть строго меньше
    assert metrics_with_fees["total_net_pnl_usdt"] < metrics_no_fees["total_net_pnl_usdt"]

    # Проверяем, что в отчете о сделках комиссии и проскальзывание не равны нулю
    out_dir_fees = metrics_with_fees["output_dir"]
    trades_log_path = os.path.join(out_dir_fees, "trades.csv")
    df_trades = pd.read_csv(trades_log_path)

    assert df_trades["commission_quote"].sum() > 0
    assert df_trades["slippage_quote"].sum() > 0

def test_leverage_effect_on_pnl(tmp_path):
    """Проверяет, что кредитное плечо корректно мультиплицирует PnL."""
    # 1. Данные и базовые параметры для простого направленного трейда
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df_market = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": [100, 110, 110], # Рост цены на 10%
        "volume": 1.0
    })
    data_csv = tmp_path / "data.csv"
    df_market.to_csv(data_csv, index=False)

    base_params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 10000.0,
        "rebalance_threshold": 0.0, # Ребаланс в начале
        "target_weights_normal": {"BTC_PERP_LONG": 1.0}, # 100% в лонг
        "data_settings": {},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True,
        "apply_signal_logic": False,
        "safe_mode_config": {"enabled": False},
        "circuit_breaker_config": {"enabled": False},
        "commission_taker": 0.0,
        "slippage_percent": 0.0,
    }

    # 2. Прогон с плечом 1x
    params_1x = copy.deepcopy(base_params)
    params_1x["futures_leverage"] = 1.0
    params_1x["report_path_prefix"] = str(tmp_path / "reports_1x")
    metrics_1x = run_backtest(params_1x, str(data_csv), is_optimizer_call=False)

    # 3. Прогон с плечом 10x
    params_10x = copy.deepcopy(base_params)
    params_10x["futures_leverage"] = 10.0
    params_10x["report_path_prefix"] = str(tmp_path / "reports_10x")
    metrics_10x = run_backtest(params_10x, str(data_csv), is_optimizer_call=False)

    # 4. Проверка
    pnl_1x = metrics_1x["total_net_pnl_usdt"]
    pnl_10x = metrics_10x["total_net_pnl_usdt"]

    # PnL должен быть примерно в 10 раз больше. 
    # В этом простом сценарии (одна позиция, без доп. ребалансов) он должен быть почти точным.
    # Начальная позиция 10000 USDT. Рост цены 10%. 
    # PnL 1x = 10000 * 1 * 0.10 = 1000
    # PnL 10x = 10000 * 10 * 0.10 = 10000
    assert pnl_1x == pytest.approx(1000.0)
    assert pnl_10x == pytest.approx(10000.0)

def test_graceful_handling_of_empty_data(tmp_path):
    """Проверяет корректную обработку пустых или невалидных данных."""
    base_params = {
        "main_asset_symbol": "BTC",
        "initial_portfolio_value_usdt": 10000.0,
        "rebalance_threshold": 0.05,
        "target_weights_normal": {"BTC_SPOT": 1.0},
        "data_settings": {},
        "report_path_prefix": str(tmp_path / "reports"),
        "use_fixed_report_path": True
    }

    # Сценарий 1: Пустой CSV-файл с рыночными данными
    empty_data_csv = tmp_path / "empty_data.csv"
    empty_data_csv.touch()
    metrics_empty = run_backtest(base_params, str(empty_data_csv))
    assert metrics_empty["status"] == "Market data empty"
    assert metrics_empty["total_net_pnl_usdt"] == 0.0

    # Сценарий 2: Пустой CSV-файл с сигналами
    valid_data_csv = tmp_path / "valid_data.csv"
    pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")], "close": [100]}).to_csv(valid_data_csv, index=False)
    empty_signals_csv = tmp_path / "empty_signals.csv"
    empty_signals_csv.touch()
    params_empty_signals = copy.deepcopy(base_params)
    params_empty_signals["data_settings"]["signals_csv_path"] = str(empty_signals_csv)
    metrics_empty_signals = run_backtest(params_empty_signals, str(valid_data_csv))
    assert metrics_empty_signals["status"] == "Completed"

    # Сценарий 3: Отсутствует колонка 'close'
    invalid_data_csv = tmp_path / "invalid_data.csv"
    pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")], "price": [100]}).to_csv(invalid_data_csv, index=False)
    with pytest.raises(KeyError):
        run_backtest(base_params, str(invalid_data_csv))

def test_main_cli_execution(tmp_path, monkeypatch):
    """Проверяет запуск бэктестера из командной строки."""
    # 1. Создаем базовые данные и конфиг
    data_csv = tmp_path / "data.csv"
    df = pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")], "close": [100]})
    df.to_csv(data_csv, index=False)

    config_path = tmp_path / "config.json"
    config = {
        "backtest_settings": {
            "main_asset_symbol": "BTC",
            "initial_portfolio_value_usdt": 1000.0,
            "rebalance_threshold": 1.0,
            "target_weights_normal": {"BTC_SPOT": 1.0},
            "data_settings": {"csv_file_path": str(data_csv)},
            "report_path_prefix": str(tmp_path / "cli_reports"),
            "use_fixed_report_path": True
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # 2. Эмулируем аргументы командной строки и запускаем main
    args = ["script_name", "--config_file", str(config_path)]
    monkeypatch.setattr('sys.argv', args)

    backtester_main()

    # 3. Проверяем, что отчеты были созданы
    report_dir = tmp_path / "cli_reports"
    assert os.path.isdir(report_dir)
    assert os.path.exists(report_dir / "summary.csv")
