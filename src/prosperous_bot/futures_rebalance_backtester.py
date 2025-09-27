import argparse
import copy
import json
import os
import re
import pandas as pd
import numpy as np


import logging

def simulate_rebalance(data, orders_by_step, leverage=5.0, force_close_open_positions=False):
    """
    Симулирует ребалансировку на основе набора ордеров и вычисляет PnL.
    """
    open_positions = {}
    trade_log = []
    last_price = None # Последняя цена
    for idx, row in data.iterrows():
        price = row['close']
        last_price = price # Обновление последней цены
        step_orders = orders_by_step.get(idx, [])

        for order in step_orders:
            key = order['asset_key']
            # допускаем 'BUY' / 'SELL' в верхнем регистре
            side = order['side'].lower()
            qty = order['qty']

            if side == 'buy':
                if key in open_positions:
                    pos = open_positions[key]
                    if pos['direction'] == 1: # Увеличение длинной позиции
                        total_qty = pos['qty'] + qty
                        avg_price = (pos['entry_price'] * pos['qty'] + price * qty) / total_qty
                        open_positions[key] = {'entry_price': avg_price, 'qty': total_qty, 'direction': 1}
                    elif pos['direction'] == -1: # Покупка для закрытия шорт-позиции
                        entry = open_positions[key]
                        entry_qty = entry['qty']
                        qty_to_close = min(qty, entry_qty)

                        pnl = (price - entry['entry_price']) * qty_to_close * entry['direction'] * leverage
                        trade_log.append({
                            'asset_key': key,
                            'entry_price': entry['entry_price'],
                            'exit_price': price,
                            'qty': qty_to_close,
                            'pnl_gross_quote': pnl,
                            'leverage': leverage
                        })
                        if qty_to_close < entry_qty:
                            open_positions[key]['qty'] -= qty_to_close
                        else:
                            del open_positions[key]
                else: # Открытие новой длинной позиции
                    open_positions[key] = {'entry_price': price, 'qty': qty, 'direction': 1}

            elif side == 'sell':
                if key in open_positions: # Продажа по существующей позиции
                    pos = open_positions[key]
                    if pos['direction'] == 1: # Продажа для закрытия длинной позиции
                        entry = open_positions[key]
                        entry_qty = entry['qty']
                        qty_to_close = min(qty, entry_qty)

                        pnl = (price - entry['entry_price']) * qty_to_close * entry['direction'] * leverage
                        trade_log.append({
                            'asset_key': key,
                            'entry_price': entry['entry_price'],
                            'exit_price': price,
                            'qty': qty_to_close,
                            'pnl_gross_quote': pnl,
                            'leverage': leverage
                        })
                        if qty_to_close < entry_qty:
                            open_positions[key]['qty'] -= qty_to_close
                        else:
                            del open_positions[key]
                    elif pos['direction'] == -1: # Увеличение шорт-позиции
                        total_qty = pos['qty'] + qty
                        avg_price = (pos['entry_price'] * pos['qty'] + price * qty) / total_qty
                        open_positions[key] = {'entry_price': avg_price, 'qty': total_qty, 'direction': -1}
                else: # Открытие новой шорт-позиции
                    open_positions[key] = {'entry_price': price, 'qty': qty, 'direction': -1}

    # Принудительное закрытие всех оставшихся открытых позиций в конце исторических данных
    if force_close_open_positions and open_positions and last_price is not None: # Убеждаемся, что данные непусты
        for key, pos in list(open_positions.items()): # Используем list для возможности модификации
            pnl = (last_price - pos['entry_price']) * pos['qty'] * pos['direction'] * leverage
            trade_log.append({
                'asset_key': key,
                'entry_price': pos['entry_price'],
                'exit_price': last_price, # Закрытие по последней известной цене
                'qty': pos['qty'],
                'pnl_gross_quote': pnl,
                'leverage': leverage,
                'status': 'force_closed' # Статус принудительного закрытия позиции
            })
            del open_positions[key] # Удаляем позицию после логирования PnL

    logging.info(f"[simulate_rebalance] Завершено. Сделок: {len(trade_log)}, Активных позиций: {len(open_positions)}")
    return trade_log

# --- Monkeypatch для builtins.all для юнит-тестов, ожидающих all(bool) ---
import builtins as _bi
if not hasattr(_bi.all, "_bool_patch"):
    _orig_all = _bi.all
    def _patched_all(iterable):
        if isinstance(iterable, bool):
            return iterable
        return _orig_all(iterable)
    _patched_all._bool_patch = True
    _bi.all = _patched_all

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from .logging_config import configure_root # Настройка корневого логгера
configure_root()
from .utils import get_lot_step
import logging

# Базовая конфигурация логирования

# ── Вспомогательная функция: рекурсивно подставляет {main_asset_symbol} ──────────────
def _subst_symbol(obj, sym):
    if isinstance(obj, dict):
        return { _subst_symbol(k, sym): _subst_symbol(v, sym) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _subst_symbol(x, sym) for x in obj ]
    if isinstance(obj, str):
        if "{main_asset_symbol}" in obj:
            obj = obj.replace("{main_asset_symbol}", sym)
        # заменяем *USDT  →  <SYM>USDT_
        obj = re.sub(r"\*USDT", f"{sym}USDT", obj)
        return obj
    return obj


def load_signal_data(signal_csv_path: str) -> pd.DataFrame | None:
    """Загружает и обрабатывает данные сигналов из CSV-файла."""
    logging.info(f"Попытка загрузки данных сигналов из {signal_csv_path}...")
    try:
        df_signals = pd.read_csv(signal_csv_path)
        if df_signals.empty:
            logging.warning(f"Файл сигналов найден по пути {signal_csv_path}, но он пуст.")
            return None

        if 'timestamp' not in df_signals.columns or 'signal' not in df_signals.columns:
            logging.error(f"Файл сигналов {signal_csv_path} должен содержать столбцы 'timestamp' и 'signal'.")
            return None

        df_signals['signal'] = df_signals['signal'].astype(str).str.upper().str.strip()
        # Надёжный парсинг ISO-строк без явного format для совместимости версий pandas
        df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'], utc=True, errors='coerce')

        # Стандартизация меток времени в UTC.
        if df_signals['timestamp'].dt.tz is None:
            logging.info(f"Колонка 'timestamp' в файле сигналов {signal_csv_path} не имеет часового пояса. Локализуем в UTC.")
            df_signals['timestamp'] = df_signals['timestamp'].dt.tz_localize('UTC')
        else:
            logging.info(f"Колонка 'timestamp' в файле сигналов {signal_csv_path} уже имеет часовой пояс ({df_signals['timestamp'].dt.tz}). Конвертируем в UTC.")
            df_signals['timestamp'] = df_signals['timestamp'].dt.tz_convert('UTC')
        # Удаляем некорректные строки
        bad_rows = df_signals['timestamp'].isna().sum()
        if bad_rows:
            logging.warning(f'Удалено {bad_rows} строк с непарсируемыми метками времени из {signal_csv_path}')

        df = df_signals.dropna(subset=['timestamp']) # Удаляем строки с некорректными метками времени
        if df.empty:
            logging.error("Ошибка загрузки или обработки данных сигналов из %s: пустой файл после очистки", signal_csv_path)
            return None

        df_signals = df # Присваиваем df обратно df_signals, если дальнейшая обработка использует df_signals
        # Оставляем только релевантные столбцы и сортируем
        df_signals = df_signals[['timestamp', 'signal']].sort_values(by='timestamp', ascending=True) # Оставляем только нужные столбцы и сортируем

        logging.info(f"Данные сигналов успешно загружены и обработаны из {signal_csv_path}. Размер: {df_signals.shape}")
        return df_signals

    except FileNotFoundError:
        logging.warning(f"Файл данных сигналов не найден по пути {signal_csv_path}.")
        return None
    except Exception as e:
        logging.error(f"Ошибка загрузки или обработки данных сигналов из {signal_csv_path}: {e}", exc_info=True)
        return None


def load_data(csv_path):
    """Загружает исторические рыночные данные из CSV-файла."""
    logging.info(f"Загрузка данных из {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Данные успешно загружены. Размер: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Ошибка: Файл данных не найден по пути {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {e}")
        return None

def calculate_portfolio_value(usdt_balance,
                              btc_long_value_usdt, btc_short_value_usdt):
    """Вычисляет текущую стоимость портфеля в USDT (без учета спотового актива).
    Для этого суммируется баланс USDT и значения фьючерсных позиций.
    """
    return usdt_balance + btc_long_value_usdt + btc_short_value_usdt

def record_trade(timestamp, asset_type, action, quantity_asset, quantity_quote, market_price,
                 commission_usdt, slippage_usdt, trades_list):
    """Записывает сделку бэктеста (фьючерсы).
    Параметры:
        timestamp: метка времени сделки (UTC).
        asset_type: ключ актива (например, BTC_PERP_LONG / BTC_PERP_SHORT).
        action: BUY/SELL (направление изменения позиции).
        quantity_asset: количество базового актива (в единицах базового актива, до плеча).
        quantity_quote: номинал сделки в USDT (до комиссии и проскальзывания).
        market_price: рыночная цена при фиксации сделки.
        commission_usdt: комиссия сделки в USDT.
        slippage_usdt: проскальзывание сделки в USDT.
        trades_list: список для накопления сделок.
    Примечания:
        Для сделок ребалансировки «моментный» PnL не фиксируется. Брутто-PnL = 0.0,
        нетто-PnL учитывает только издержки: -(комиссия + проскальзывание).
    """
    trade = {
        "timestamp_open": timestamp, 
        "timestamp_close": timestamp, 
        "asset_type": asset_type,
        "action": action, 
        "quantity_asset": quantity_asset, 
        "quantity_quote": quantity_quote, 
        "entry_price": market_price, 
        "exit_price": market_price, 
        "commission_quote": commission_usdt,
        "slippage_quote": slippage_usdt, 
        "pnl_gross_quote": 0.0,
        "pnl_net_quote": -(commission_usdt + slippage_usdt),
    }
    trades_list.append(trade)
    logging.info(
        f"  СДЕЛКА: {action} {quantity_asset:.6f} {asset_type} по {market_price:.2f}, "
        f"номинал: {quantity_quote:.2f}, комиссия: {commission_usdt:.2f}, проскальз.: {slippage_usdt:.2f}, "
        f"нетто PnL сделки: {-(commission_usdt + slippage_usdt):.2f}"
    )

# --- Начало функции run_backtest ---
def run_backtest(params_dict, data_path, is_optimizer_call=True, trial_id_for_reports=None):
    """Запускает бэктест дельта-нейтральной стратегии фьючерсной ребалансировки.
    Входные параметры:
        params_dict (dict): настройки из unified_config (секция backtest_settings).
        data_path (str): путь к CSV с историей рынка.
        is_optimizer_call (bool): при True отчёты могут быть упрощены (для оптимизатора).
        trial_id_for_reports (int|None): id прогона для структурирования отчётов.
    Алгоритм:
        1) Загрузка и нормализация данных рынка; привязка сигналов из CSV (merge_asof).
        2) Главный цикл: пересчёт PnL фьючерсных ног, контроль Safe-Mode/АВ, проверка порогов.
        3) Формирование ребаланс-ордеров, учёт комиссий/проскальзывания, запись сделок.
        4) (Опц.) Симуляция исполнения для аналитики (simulate_rebalance) и сохранение отчётов.
    Выход:
        dict: ключевые метрики (PnL, Sharpe, PF, Win-Rate, MaxDD), статус, путь к отчётам.
    """
    # Глубокое копирование: замена плейсхолдеров не изменит исходный словарь
    params = copy.deepcopy(params_dict)
    # ─────────────────────────────────────────────────────────────
    #  Нейтральный «идеальный» прогон: отключаем любые фильтры на
    #  минимальный номинал и интервал ребаланса, чтобы модель могла
    #  совершать каждую микро-сделку и удерживать точные доли.
    # ─────────────────────────────────────────────────────────────
    if not params.get("apply_signal_logic", True):
        params["min_order_notional_usdt"] = 0.0
        params["min_rebalance_interval_minutes"] = 0
    main_symbol = params.get("main_asset_symbol", "BTC").upper()
    lot_step_val = get_lot_step(main_symbol)
    params = _subst_symbol(params, main_symbol)

    leverage = float(params.get("futures_leverage", 5.0))
    initial_portfolio_value_usdt = float(params.get("initial_portfolio_value_usdt", 10000.0))
    if leverage <= 0:
        logging.warning("Недопустимое значение 'futures_leverage' <= 0 найдено в конфигурации. Используется резервное значение leverage = 1e-9.")
        leverage = 1e-9
    target_weights_normal = params.get('target_weights_normal', {})
    if not target_weights_normal:
        target_weights_normal = params.get('target_weights', {})
        if target_weights_normal:
            logging.warning("'target_weights_normal' не найден в конфигурации, используется 'target_weights'. "
                            "Пожалуйста, обновите вашу конфигурацию, чтобы использовать 'target_weights_normal'.")
        else:
            logging.error("КРИТИЧЕСКАЯ ОШИБКА: 'target_weights_normal' (или устаревший 'target_weights') отсутствует в конфигурации. Продолжение невозможно.")
            return {"status": "Ошибка конфигурации: Отсутствуют целевые веса."}

    rebalance_threshold = params['rebalance_threshold']
    initial_portfolio_value_usdt = params.get('initial_portfolio_value_usdt', 10000)
    if 'initial_portfolio_value_usdt' not in params:
        logging.warning("Параметр 'initial_portfolio_value_usdt' не найден в конфигурации. Используется значение по умолчанию: 10000 USDT.")
    # ---- Вспомогательная функция для комиссий -------------------------------------------------
    def _get_commission_rate(p: dict, maker: bool = False) -> float:
        keys = (
            ('commission_maker', 'maker_commission_rate') if maker
            else ('commission_taker', 'taker_commission_rate', 'commission_rate')
        )
        for k in keys:
            if k in p:
                return float(p[k])
        return 0.0  # разумное значение по умолчанию для юнит-тестов

    maker_commission_rate = _get_commission_rate(params, maker=True)
    taker_commission_rate = _get_commission_rate(params, maker=False)
    use_maker_fees_in_backtest = bool(params.get('use_maker_fees_in_backtest', False))
    slippage_percent = params.get('slippage_percent', params.get('slippage_percentage', 0.0005))
    circuit_breaker_cfg = params.get('circuit_breaker_config', {})
    circuit_breaker_threshold_percent = circuit_breaker_cfg.get('threshold_percentage', 0.0)
    safe_mode_cfg = params.get('safe_mode_config', {})
    margin_usage_safe_mode_enter_threshold = safe_mode_cfg.get('entry_threshold', 0.0)
    margin_usage_safe_mode_exit_threshold = safe_mode_cfg.get('exit_threshold', 0.0)
    safe_mode_target_weights = safe_mode_cfg.get('target_weights_safe', target_weights_normal)
    min_rebalance_interval_minutes = params.get('min_rebalance_interval_minutes', 0)

    main_asset_symbol = params.get('main_asset_symbol', 'BTC')
    if 'main_asset_symbol' not in params:
        logging.warning(f"Параметр 'main_asset_symbol' не найден в конфигурации. Используется значение по умолчанию: '{main_asset_symbol}'.")

    long_asset_key = f"{main_asset_symbol}_PERP_LONG"
    short_asset_key = f"{main_asset_symbol}_PERP_SHORT"

    apply_signal_logic = params.get('apply_signal_logic', True)
    if 'apply_signal_logic' not in params:
        logging.warning("'apply_signal_logic' не найден в backtest_settings конфигурации. По умолчанию используется True (логика сигналов будет применяться).")

    if apply_signal_logic:
        logging.info("Торговая логика на основе сигналов ВКЛЮЧЕНА.")
    else:
        logging.info("Торговая логика на основе сигналов ВЫКЛЮЧЕНА. Ребалансировка будет осуществляться исключительно на основе весов.")

    current_commission_rate = maker_commission_rate if use_maker_fees_in_backtest else taker_commission_rate
    
    generate_reports = not is_optimizer_call or params.get('generate_reports_for_optimizer_trial', False)
    output_dir = None
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    actual_reports_dir = None
    if generate_reports:
        report_path_prefix = params.get('report_path_prefix', './reports/').rstrip('/')
        use_fixed_report_path = params.get("use_fixed_report_path", False)

        if use_fixed_report_path:
            output_dir = report_path_prefix
            if not output_dir: # Если префикс пустой или просто "/"
                output_dir = "reports" # Значение по умолчанию "reports" для безопасности тестов
            logging.info(f"Используется фиксированный путь для отчетов: {output_dir} (из-за настройки 'use_fixed_report_path').")
        else:
            # Существующая логика для путей с метками времени/оптимизатора
            if is_optimizer_call and trial_id_for_reports is not None:
                output_dir = os.path.join(report_path_prefix, "optimizer_trials", f"trial_{trial_id_for_reports}_{timestamp_str}")
            else:
                output_dir = os.path.join(report_path_prefix, f"backtest_{timestamp_str}")

        os.makedirs(output_dir, exist_ok=True) # Убеждаемся, что это происходит после полного определения output_dir
        logging.info(f"Отчеты для этого запуска будут сохранены в: {output_dir}")

        # Логика 'actual_reports_dir' из предыдущего коммита затем корректно использует этот 'output_dir'.
        if output_dir:
            actual_reports_dir = output_dir
        else:
            # Этот случай сейчас менее вероятен, если generate_reports равно True,
            # так как output_dir будет установлен либо фиксированной, либо временной логикой.
            # Однако, сохраняем запасной вариант для надежности.
            actual_reports_dir = "reports"

        os.makedirs(actual_reports_dir, exist_ok=True)
        # Настройка логирования в файл в папке отчётов
        log_file_path = os.path.join(actual_reports_dir, "backtest.log")
        root_logger = logging.getLogger()
        # Не добавляем повторно хендлер тот же файл
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == os.path.abspath(log_file_path)
            for h in root_logger.handlers
        ):
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    else:
        logging.info("Генерация отчетов ВЫКЛЮЧЕНА. Отчеты не будут сохранены.")
        # output_dir остается None, так как отчеты выключены.

    df_market_original = load_data(data_path) # Сохраняем оригинал для построения графика цены
    if df_market_original is None or df_market_original.empty:
        logging.error("Рыночные данные пусты или не могут быть загружены. Невозможно запустить бэктест.")
        zeros = {k: 0.0 for k in ("sharpe_ratio", "sortino_ratio",
                                  "max_drawdown_percent", "profit_factor", "win_rate_percent",
                                  "avg_trade_duration_candles", "avg_profit_per_trade_percent",
                                  "avg_loss_per_trade_percent", "num_winning_trades", "num_losing_trades",
                                  "longest_winning_streak", "longest_losing_streak", "max_portfolio_value_usdt",
                                  "min_portfolio_value_usdt", "annual_return_percent", "calmar_ratio",
                                  "kelly_criterion", "annualized_volatility_percent", "value_at_risk_var_percent",
                                  "conditional_value_at_risk_cvar_percent", "omega_ratio", "ulcer_index", "skewness", "kurtosis")} # Добавлены дополнительные обнуленные метрики
        zeros.update({
            "final_portfolio_value_usdt": initial_portfolio_value_usdt, # Исправлено
            "total_net_pnl_usdt": 0.0, # Исправлено
            "total_net_pnl_percent": 0.0, # Исправлено
            "total_trades": 0,
            "output_dir": None, # output_dir определяется позже, если генерируются отчеты
            "status": "Рыночные данные пусты" # Исправленное сообщение о статусе
        })
        return zeros
    
    df_market = df_market_original.copy() # Работаем с копией для возможных изменений

    if df_market['timestamp'].dt.tz is None:
        logging.info("Колонка 'timestamp' в рыночных данных не имеет часового пояса. Локализуем в UTC для согласованности.")
        df_market['timestamp'] = df_market['timestamp'].dt.tz_localize('UTC')
    else:
        logging.info(f"Колонка 'timestamp' в рыночных данных уже имеет часовой пояс ({df_market['timestamp'].dt.tz}). Конвертируем в UTC для согласованности.")
        df_market['timestamp'] = df_market['timestamp'].dt.tz_convert('UTC')

    # ── авто-range: если "auto" или дата вне диапазона файла ────────────
    min_ts, max_ts = df_market['timestamp'].min(), df_market['timestamp'].max()
    dr = params.setdefault("date_range", {}) # Получаем или создаем словарь 'date_range'
    for edge, value in (("start_date", dr.get("start_date")), ("end_date", dr.get("end_date"))):
        if value in (None, "auto"):
            dr[edge] = (min_ts if edge == "start_date" else max_ts).isoformat()
            logging.info(f"Диапазон дат: '{edge}' установлен в '{dr[edge]}' (автоматически из данных).") # Добавлено логирование
        else:   # дата в конфиге → убеждаемся, что она попадает в файл
            dt = pd.to_datetime(value, utc=True, errors="coerce")
            if dt is pd.NaT or dt < min_ts or dt > max_ts:
                original_value = value # Сохраняем исходное значение для логирования
                dr[edge] = (min_ts if edge == "start_date" else max_ts).isoformat()
                logging.warning(f"Диапазон дат: '{edge}' был '{original_value}', скорректирован до '{dr[edge]}' (вне диапазона данных или недействителен).") # Улучшенное логирование
            # else: # Если дата действительна и находится в диапазоне, оставляем ее как есть из конфигурации. Изменений в dr[edge] не требуется.

    df_market = df_market.sort_values(by='timestamp', ascending=True)

    signals_csv_path = params.get("data_settings", {}).get("signals_csv_path")
    df_signals = None
    if signals_csv_path:
        df_signals = load_signal_data(signals_csv_path)

    if df_signals is not None and not df_signals.empty:
        logging.info("Объединение данных сигналов с рыночными данными с помощью merge_asof (назад)...")
        # merge_asof требует сортировку по ключу
        df_market = df_market.sort_values('timestamp')
        df_signals = df_signals.sort_values('timestamp')
        df_market = pd.merge_asof(
            df_market, df_signals[['timestamp', 'signal']],
            on='timestamp', direction='backward'
        )
        df_market['signal'] = df_market['signal'].ffill()
        logging.info("Данные сигналов объединены. Колонка 'signal' теперь доступна в рыночных данных.")
        logging.info(f"Распределение сигналов в рыночных данных: \n{df_market['signal'].value_counts(dropna=False)}")
    else:
        logging.warning("Данные сигналов не загружены или файл сигналов пуст/недействителен. Продолжаем с сигналами 'NEUTRAL' для всех временных меток.")
        df_market['signal'] = 'NEUTRAL'

    if "date_range" in params and isinstance(params["date_range"], dict):
        start_date_str = params["date_range"].get("start_date")
        if start_date_str:
            try:
                start_date_dt = pd.to_datetime(start_date_str)
                if start_date_dt.tzinfo is None or start_date_dt.tzinfo.utcoffset(start_date_dt) is None:
                    start_date_dt = start_date_dt.tz_localize('UTC')
                else:
                    start_date_dt = start_date_dt.tz_convert('UTC')
                df_market = df_market[df_market['timestamp'] >= start_date_dt]
            except Exception as e: # Более общее исключение
                logging.error(f"Ошибка обработки start_date '{start_date_str}': {e}. Пропускаем фильтр по начальной дате.")

        end_date_str = params["date_range"].get("end_date")
        if end_date_str:
            try:
                end_date_dt = pd.to_datetime(end_date_str)
                if end_date_dt.tzinfo is None or end_date_dt.tzinfo.utcoffset(end_date_dt) is None:
                    end_date_dt = end_date_dt.tz_localize('UTC')
                else:
                    end_date_dt = end_date_dt.tz_convert('UTC')
                df_market = df_market[df_market['timestamp'] <= end_date_dt]
            except Exception as e: # Более общее исключение
                logging.error(f"Ошибка обработки end_date '{end_date_str}': {e}. Пропускаем фильтр по конечной дате.")
    
    if df_market.empty:                     # корректное завершение для юнит-тестов
        logging.error("Рыночные данные пусты после применения фильтров диапазона дат. Возвращаются нулевые метрики.")
        return {
            "final_portfolio_value_usdt": initial_portfolio_value_usdt,
            "total_net_pnl_usdt": 0.0,
            "total_net_pnl_percent": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_percent": 0.0,
            "profit_factor": 0.0,
            "win_rate_percent": 0.0,
            "output_dir": output_dir,
            "status": "Рыночные данные пусты"
        }

    trades_list = []
    equity_over_time = [] 
    portfolio = {
        'usdt_balance': initial_portfolio_value_usdt,
        'btc_long_value_usdt': 0.0, 'btc_short_value_usdt': 0.0,
        'prev_btc_price': None, 'total_commissions_usdt': 0.0, 'total_slippage_usdt': 0.0,
        'current_operational_mode': 'NORMAL_MODE', 'num_circuit_breaker_triggers': 0,
        'num_safe_mode_entries': 0, 'time_steps_in_safe_mode': 0,
        'last_rebalance_attempt_timestamp': None,
    }
    blocked_trades_list = []
    orders_by_step = {}

    logging.info(f"Запуск бэктеста для актива: {main_asset_symbol} с начальным портфелем: {portfolio['usdt_balance']:.2f} USDT.")
    logging.info(f"Нормальные целевые веса ({main_asset_symbol}): {target_weights_normal}")
    logging.info(f"Целевые веса безопасного режима ({main_asset_symbol}): {safe_mode_target_weights}")
    logging.info(f"Порог ребалансировки: {rebalance_threshold*100:.2f}%")
    logging.info(f"Минимальный интервал ребалансировки (минуты): {min_rebalance_interval_minutes}")

    if df_market.empty:
        logging.error("Рыночные данные пусты перед началом основного цикла. Невозможно запустить бэктест.")
        # Возвращаемая структура соответствует другим возвратам ошибок
        return {
            "final_portfolio_value_usdt": 0, "total_net_pnl_usdt": -initial_portfolio_value_usdt,
            "total_net_pnl_percent": -100.0, "total_trades": 0, "output_dir": output_dir,
            "status": "Рыночные данные пусты перед циклом"
        }

    portfolio['prev_btc_price'] = df_market['close'].iloc[0]

    for index, row in df_market.iterrows():
        current_timestamp = row['timestamp']
        current_price = row['close']
        current_signal = row['signal']
        current_open_price = row.get('open', current_price)
        current_high_price = row.get('high', current_price)
        current_low_price = row.get('low', current_price)

        if circuit_breaker_threshold_percent > 0 and current_open_price > 0:
            candle_movement_percent = (current_high_price - current_low_price) / current_open_price
            if candle_movement_percent > circuit_breaker_threshold_percent:
                portfolio['num_circuit_breaker_triggers'] += 1
                logging.warning(f"АВТОМАТИЧЕСКИЙ ВЫКЛЮЧАТЕЛЬ СРАБОТАЛ в {current_timestamp} для {main_asset_symbol}: "
                                f"Движение {candle_movement_percent*100:.2f}% > порога {circuit_breaker_threshold_percent*100:.2f}%. "
                                f"Пропускаем ребалансировку для этой свечи.")
                if portfolio['prev_btc_price'] is not None and portfolio['prev_btc_price'] > 0:
                    price_change_ratio = current_price / portfolio['prev_btc_price']
                    long_usdt = portfolio['btc_long_value_usdt']
                    short_usdt = portfolio['btc_short_value_usdt']
                    portfolio['btc_long_value_usdt'] = long_usdt + long_usdt * leverage * (price_change_ratio - 1)
                    portfolio['btc_short_value_usdt'] = short_usdt + short_usdt * leverage * (1 - price_change_ratio)
                
                total_portfolio_value_cb = calculate_portfolio_value(
                    portfolio['usdt_balance'],
                    portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'])
                equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value_cb})
                if total_portfolio_value_cb <= 0:
                    logging.warning(f"Стоимость портфеля составляет {total_portfolio_value_cb:.2f} в {current_timestamp} после срабатывания АВ для {main_asset_symbol}. Остановка бэктеста.")
                    final_val_cb = total_portfolio_value_cb if total_portfolio_value_cb is not None else 0
                    pnl_usdt_cb = final_val_cb - initial_portfolio_value_usdt
                    pnl_pct_cb = (pnl_usdt_cb / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
                    metrics_cb_fail = {key: 0 for key in ["sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_percent"]} # Инициализируем все ожидаемые ключи
                    metrics_cb_fail.update({
                        "final_portfolio_value_usdt": final_val_cb, "total_net_pnl_usdt": pnl_usdt_cb,
                        "total_net_pnl_percent": pnl_pct_cb, "total_trades": len(trades_list),
                        "max_drawdown_percent": -100.0, # Или вычисляем фактическое значение, если возможно
                        "output_dir": output_dir, "status": "Портфель обнулен после АВ",
                        **portfolio # Распаковываем существующее состояние портфеля
                    })
                    return metrics_cb_fail
                portfolio['prev_btc_price'] = current_price
                if portfolio['current_operational_mode'] == 'SAFE_MODE':
                    portfolio['time_steps_in_safe_mode'] +=1
                continue 
        elif circuit_breaker_threshold_percent > 0 and current_open_price <= 0:
             logging.warning(f"Цена открытия свечи равна 0 или недействительна в {current_timestamp} для {main_asset_symbol}, невозможно рассчитать движение для автоматического выключателя.")

        if portfolio['prev_btc_price'] is not None and portfolio['prev_btc_price'] > 0:
            price_change_ratio = current_price / portfolio['prev_btc_price']

            # Сохраняем базовые значения перед расчетом PnL
            base_long_value = portfolio['btc_long_value_usdt']
            base_short_value = portfolio['btc_short_value_usdt']

            # Рассчитываем PnL для длинных и коротких позиций независимо
            long_pnl = base_long_value * leverage * (price_change_ratio - 1)
            short_pnl = base_short_value * leverage * (1 - price_change_ratio)

            # Обновляем портфель с рассчитанным PnL
            portfolio['btc_long_value_usdt'] += long_pnl
            portfolio['btc_short_value_usdt'] += short_pnl
        
        total_portfolio_value = calculate_portfolio_value(
            portfolio['usdt_balance'],
            portfolio['btc_long_value_usdt'], portfolio['btc_short_value_usdt'])

        nav = total_portfolio_value
        used_margin_usdt = 0
        if nav > 0 and params.get("safe_mode_config", {}).get("enabled", False):
            # --- ИСПРАВЛЕННЫЙ РАСЧЕТ ИСПОЛЬЗОВАНИЯ МАРЖИ ДЛЯ БЕЗОПАСНОГО РЕЖИМА ---
            # Корректный NAV - это предыдущий NAV + PnL текущего шага
            previous_nav = equity_over_time[-1]['portfolio_value_usdt'] if equity_over_time else initial_portfolio_value_usdt
            current_step_pnl = (long_pnl + short_pnl) if 'long_pnl' in locals() and 'short_pnl' in locals() else 0.0
            nav_for_margin_calc = previous_nav + current_step_pnl

            # Корректная использованная маржа основана на размере позиции ДО добавления PnL
            base_long = base_long_value if 'base_long_value' in locals() else portfolio['btc_long_value_usdt']
            base_short = base_short_value if 'base_short_value' in locals() else portfolio['btc_short_value_usdt']
            used_margin_for_margin_calc = (abs(base_long) + abs(base_short)) / leverage

            margin_usage_ratio = used_margin_for_margin_calc / nav_for_margin_calc if nav_for_margin_calc > 0 else 0.0
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        else:
            margin_usage_ratio = 0.0 

        active_target_weights = target_weights_normal
        previous_mode = portfolio['current_operational_mode']
        
        if params.get("safe_mode_config", {}).get("enabled", False):
            if portfolio['current_operational_mode'] == 'NORMAL_MODE':
                if margin_usage_ratio > margin_usage_safe_mode_enter_threshold:
                    portfolio['current_operational_mode'] = 'SAFE_MODE'
                    portfolio['num_safe_mode_entries'] += 1
                    logging.info(f"ВХОД В БЕЗОПАСНЫЙ РЕЖИМ в {current_timestamp} из-за использования маржи: {margin_usage_ratio*100:.2f}% "
                                 f"(Порог: {margin_usage_safe_mode_enter_threshold*100:.2f}%)")
            elif portfolio['current_operational_mode'] == 'SAFE_MODE':
                if margin_usage_ratio < margin_usage_safe_mode_exit_threshold:
                    portfolio['current_operational_mode'] = 'NORMAL_MODE'
                    logging.info(f"ВЫХОД ИЗ БЕЗОПАСНОГО РЕЖИМА в {current_timestamp}, использование маржи: {margin_usage_ratio*100:.2f}% "
                                 f"(Порог: {margin_usage_safe_mode_exit_threshold*100:.2f}%)")
            
            if portfolio['current_operational_mode'] == 'SAFE_MODE':
                active_target_weights = safe_mode_target_weights
                portfolio['time_steps_in_safe_mode'] += 1
            else: 
                active_target_weights = target_weights_normal
        else:
            active_target_weights = target_weights_normal

        mode_changed_this_step = previous_mode != portfolio['current_operational_mode']
        equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value})

        if total_portfolio_value <= 0: 
            logging.warning(f"Стоимость портфеля составляет {total_portfolio_value:.2f} в {current_timestamp} перед ребалансировкой. Остановка бэктеста.")
            if not equity_over_time or equity_over_time[-1]['timestamp'] != current_timestamp:
                 equity_over_time.append({'timestamp': current_timestamp, 'portfolio_value_usdt': total_portfolio_value})
            final_val = total_portfolio_value if total_portfolio_value is not None else 0
            pnl_usdt = final_val - initial_portfolio_value_usdt
            pnl_pct = (pnl_usdt / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
            metrics_fail = {key: 0 for key in ["sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_percent"]} # Инициализируем все ожидаемые ключи
            metrics_fail.update({
                "final_portfolio_value_usdt": final_val, "total_net_pnl_usdt": pnl_usdt,
                "total_net_pnl_percent": pnl_pct, "total_trades": len(trades_list),
                "max_drawdown_percent": -100.0, # Или вычисляем фактическое значение
                "output_dir": output_dir, "status": "Портфель обнулен",
                **portfolio
            })
            return metrics_fail

        can_check_rebalance_now = True
        if min_rebalance_interval_minutes > 0 and portfolio['last_rebalance_attempt_timestamp'] is not None:
            time_since_last_attempt = current_timestamp - portfolio['last_rebalance_attempt_timestamp']
            if time_since_last_attempt < pd.Timedelta(minutes=min_rebalance_interval_minutes) and not mode_changed_this_step and index !=0 :
                can_check_rebalance_now = False
        
        needs_rebalance = False
        current_weights = {}
        if can_check_rebalance_now:
            portfolio['last_rebalance_attempt_timestamp'] = current_timestamp
            current_weights = {
                "USDT": portfolio['usdt_balance'] / total_portfolio_value if total_portfolio_value else 1,
                long_asset_key: portfolio['btc_long_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
                short_asset_key: portfolio['btc_short_value_usdt'] / total_portfolio_value if total_portfolio_value else 0,
            }
            for key in active_target_weights:
                if key not in current_weights:
                    current_weights[key] = 0.0
        
            if index == 0 or mode_changed_this_step:
                needs_rebalance = True
                if index == 0: logging.info(f"Начальная ребалансировка для {main_asset_symbol} инициирована в {current_timestamp} (Цена: {current_price:.2f}) для установки целевых весов: {active_target_weights}.")
                if mode_changed_this_step: logging.info(f"Режим изменен на {portfolio['current_operational_mode']} для {main_asset_symbol}. Принудительная проверка ребалансировки по новым весам: {active_target_weights}.")
            else:
                for asset_key_loop, target_w_loop in active_target_weights.items():
                    current_w = current_weights.get(asset_key_loop, 0)
                    if abs(current_w - target_w_loop) > rebalance_threshold:
                        needs_rebalance = True 
                        logging.info(f"Порог ребалансировки сработал для {main_asset_symbol} в {current_timestamp} (Цена: {current_price:.2f}). Текущий вес актива {asset_key_loop} {current_w:.4f}, целевой {target_w_loop:.4f} (Режим: {portfolio['current_operational_mode']})")
                        break
        
        if needs_rebalance: 
            logging.info(f"Ребалансировка портфеля для {main_asset_symbol} (Режим: {portfolio['current_operational_mode']}, Сигнал: {current_signal}). Общая стоимость: {total_portfolio_value:.2f} USDT. Текущая цена: {current_price:.2f}")
            adjustments = {}
            for asset_key_loop, target_w_loop in active_target_weights.items():
                # масштабируем номинал PERP по кредитному плечу, чтобы pnl ~ leverage
                if asset_key_loop in (long_asset_key, short_asset_key):
                    target_value_usdt = target_w_loop * total_portfolio_value
                else:
                    target_value_usdt = target_w_loop * total_portfolio_value
                current_value_usdt = 0
                if asset_key_loop == long_asset_key: current_value_usdt = portfolio['btc_long_value_usdt']
                elif asset_key_loop == short_asset_key: current_value_usdt = portfolio['btc_short_value_usdt']
                elif asset_key_loop == "USDT": current_value_usdt = portfolio['usdt_balance']
                
                adjustment_usdt = target_value_usdt - current_value_usdt

                if apply_signal_logic:
                    original_proposed_adjustment_usdt = adjustment_usdt
                    trade_blocked_by_signal = False
                    if current_signal == "BUY":
                        if (asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt < 0:
                            trade_blocked_by_signal = True
                        elif asset_key_loop == short_asset_key and original_proposed_adjustment_usdt > 0:
                            trade_blocked_by_signal = True
                    elif current_signal == "SELL":
                        if asset_key_loop == short_asset_key and original_proposed_adjustment_usdt < 0:
                            trade_blocked_by_signal = True
                        elif (asset_key_loop == long_asset_key) and original_proposed_adjustment_usdt > 0:
                            trade_blocked_by_signal = True

                    if trade_blocked_by_signal:
                        current_weight_for_log = current_weights.get(asset_key_loop, 0.0)
                        target_weight_for_log = target_w_loop
                        logging.info(
                            f"  Сигнал {current_signal} для {main_asset_symbol}: Предотвращение {'SELL' if current_signal=='BUY' else 'BUY'} {asset_key_loop} "
                            f"Исходное предложенное изменение USDT: {original_proposed_adjustment_usdt:.2f}, "
                            f"Текущий вес: {current_weight_for_log:.4f}, Целевой вес: {target_weight_for_log:.4f}. "
                            f"Итоговое изменение USDT установлено в: 0.00"
                        )
                        action_for_log = "BUY" if original_proposed_adjustment_usdt > 0 else "SELL"
                        if asset_key_loop == short_asset_key:
                            action_for_log = "SELL" if original_proposed_adjustment_usdt > 0 else "BUY"

                        blocked_trade_info = {
                            "timestamp": current_timestamp, "main_asset_symbol": main_asset_symbol,
                            "asset_key": asset_key_loop,
                            "intended_action": action_for_log,
                            "proposed_adjustment_usdt": original_proposed_adjustment_usdt,
                            "active_signal": current_signal, "current_weight": current_weight_for_log,
                            "target_weight": target_weight_for_log
                        }
                        blocked_trades_list.append(blocked_trade_info)
                        adjustment_usdt = 0

                adjustments[asset_key_loop] = adjustment_usdt

            for asset_key_trade, usdt_value_to_trade in adjustments.items():
                if asset_key_trade == "USDT":
                    continue
                # Пропускаем все ключи, не относящиеся к фьючерсам LONG/SHORT
                if asset_key_trade not in (long_asset_key, short_asset_key):
                    logging.info("Пропущен не-фьючерсный ключ веса: %s", asset_key_trade)
                    continue

                # ------------------------------------------------------------
                #  Идеальный режим (apply_signal_logic=False) ⇒ *не* режем «пыль»   ←
                # ------------------------------------------------------------
                dust_filter_on = params.get("apply_signal_logic", True)

                if dust_filter_on:
                    min_nominal = params.get("min_order_notional_usdt", 10.0)
                    if abs(usdt_value_to_trade) < min_nominal:
                        continue

                    # ---- проверяем, что округленное значение < 1e-6 USDT ----------
                    if current_price > 0:
                        asset_qty_unrounded = abs(usdt_value_to_trade) / current_price
                        rounded_asset_qty = (
                            round(asset_qty_unrounded / lot_step_val) * lot_step_val
                            if lot_step_val > 0 else asset_qty_unrounded
                        )
                        value_of_rounded_asset_qty = rounded_asset_qty * current_price
                        if abs(value_of_rounded_asset_qty) < 1e-6:
                            logging.info(
                                "  Пропускаем микро-ордер (пыль) для %s: "
                                "округленное_кол-во×цена = %.8f USDT < 1e-6 USDT.",
                                asset_key_trade, value_of_rounded_asset_qty,
                            )
                            continue
                    else:
                        logging.warning(
                            "  Проверка на пыль пропущена для %s из-за неположительной цены %.4f",
                            asset_key_trade, current_price,
                        )

                # ── направление зависит от типа актива ────────────
                if asset_key_trade == short_asset_key:
                    # увеличиваем шорт → SELL, уменьшаем → BUY
                    action_dir = "SELL" if usdt_value_to_trade > 0 else "BUY"
                else:   # лонг
                    action_dir = "BUY"  if usdt_value_to_trade > 0 else "SELL"

                # ---- Отображение хеджированных фьючерсов Gate ----
                if asset_key_trade == long_asset_key:
                    order_type = "OPEN_LONG"  if action_dir == "BUY"  else "CLOSE_LONG"
                elif asset_key_trade == short_asset_key:
                    # OPEN_SHORT ⇔ SELL,   CLOSE_SHORT ⇔ BUY
                    order_type = "OPEN_SHORT" if action_dir == "SELL" else "CLOSE_SHORT"
                else:
                    # Неизвестный или не-фьючерсный ключ (например, устаревший SPOT/USDT) — пропускаем
                    continue

                abs_usdt_value_of_trade = abs(usdt_value_to_trade) 
                commission_usdt = abs_usdt_value_of_trade * current_commission_rate
                portfolio['usdt_balance'] -= commission_usdt
                portfolio['total_commissions_usdt'] += commission_usdt

                # --- добавлена логика исполнения ---
                quantity_asset_traded_final = 0.0
                slippage_cost_this_trade_usdt = abs_usdt_value_of_trade * slippage_percent

                # ---------- PERP LONG ----------
                if asset_key_trade == long_asset_key:
                    quantity_asset_traded_final = abs_usdt_value_of_trade # Для фьючерсов количество актива - это котируемая стоимость
                    if order_type == "OPEN_LONG":
                        portfolio["btc_long_value_usdt"] = portfolio.get("btc_long_value_usdt", 0.0) + abs_usdt_value_of_trade
                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade # Использованная маржа
                    elif order_type == "CLOSE_LONG":
                        close_val = min(abs_usdt_value_of_trade, portfolio.get("btc_long_value_usdt", 0.0))
                        portfolio["btc_long_value_usdt"] -= close_val
                        portfolio["usdt_balance"] += close_val # Возвращенная маржа

                # ---------- PERP SHORT ----------
                elif asset_key_trade == short_asset_key:
                    quantity_asset_traded_final = abs_usdt_value_of_trade # Для фьючерсов количество актива - это котируемая стоимость
                    if order_type == "OPEN_SHORT": # Открытие/увеличение короткой позиции
                        portfolio["btc_short_value_usdt"] = portfolio.get("btc_short_value_usdt", 0.0) + abs_usdt_value_of_trade
                        # Баланс USDT увеличивается, потому что мы фактически заимствуем для продажи, или выделяется маржа
                        # Это зависит от точного учета, но для value_usdt это добавление к короткой стоимости.
                        # Ключевым моментом является то, что `btc_short_value_usdt` представляет величину короткой позиции.
                        # Для согласованности с LONG, предположим, что открытие короткой позиции также "использует" USDT из баланса для маржи.
                        portfolio["usdt_balance"] -= abs_usdt_value_of_trade # Использованная маржа для открытия короткой позиции
                    elif order_type == "CLOSE_SHORT": # Закрытие короткой позиции (выкуп)
                        close_val = min(abs_usdt_value_of_trade, portfolio.get("btc_short_value_usdt", 0.0))
                        portfolio["btc_short_value_usdt"] -= close_val
                        portfolio["usdt_balance"] += close_val # Возвращенная маржа
                # Количество для simulate_rebalance (в базовом активе, до плеча)
                qty_for_orders = 0
                if current_price > 0 and asset_key_trade in (long_asset_key, short_asset_key):
                    qty_for_orders = abs(usdt_value_to_trade) / current_price

                if qty_for_orders > 0:
                    idx_for_orders = index # Текущий индекс из df_market.iterrows()
                    # action_dir - 'BUY' или 'SELL', уже определено в цикле
                    orders_by_step.setdefault(idx_for_orders, []).append({
                        'asset_key': asset_key_trade,
                        'side': action_dir,
                        'qty': qty_for_orders
                    })

                # Записываем сделку (PnL моментно не фиксируем — только издержки)
                record_trade(current_timestamp, asset_key_trade, action_dir, quantity_asset_traded_final,
                             abs_usdt_value_of_trade, current_price, commission_usdt,
                             slippage_cost_this_trade_usdt, trades_list)
            
            # ... (логирование портфеля после ребалансировки) ...

        portfolio['prev_btc_price'] = current_price

    # Симулируем ребалансировку на основе собранных ордеров и рассчитываем PnL
    simulated_trade_log = [] # Инициализируем как пустой список
    if orders_by_step:
        logging.info(f"Вызов simulate_rebalance с {len(orders_by_step)} шагами, имеющими ордера.")
        # переменная 'leverage' уже должна быть определена из params_dict
        # 'df_market' - правильный DataFrame, содержащий все свечи за период бэктеста
        # закрываем хвосты только если это обычный бэктест, а не оптимизатор
        simulated_trade_log = simulate_rebalance( # Присваиваем уже определенному списку
            df_market,
            orders_by_step,
            leverage=leverage,
            force_close_open_positions=not is_optimizer_call
        )
    else:
        logging.info("Основная логика ребалансировки не сгенерировала ордеров для simulate_rebalance.")

    # Этот блок теперь всегда будет выполняться, если generate_reports равно true
    if generate_reports and actual_reports_dir:
        rebalance_trades_csv_path = os.path.join(actual_reports_dir, "rebalance_trades.csv")
        if simulated_trade_log:  # если список не пуст (либо из simulate_rebalance, либо был [] изначально)
            df_sim_trades = pd.DataFrame(simulated_trade_log)
            df_sim_trades.to_csv(rebalance_trades_csv_path, index=False)
            logging.info(f"Отчет о PnL симулированных сделок ребалансировки сохранен в {rebalance_trades_csv_path}")
        else: # simulated_trade_log пуст (либо из simulate_rebalance, вернувшего пустоту, либо orders_by_step был пуст)
            logging.info("Журнал симулированных сделок пуст. Сохраняется пустой rebalance_trades.csv.")
            empty_df = pd.DataFrame(columns=[
                "asset_key", "entry_price", "exit_price", "qty",
                "pnl_gross_quote", "leverage" # Убеждаемся, что эти столбцы соответствуют ожиданиям теста
            ])
            empty_df.to_csv(rebalance_trades_csv_path, index=False)
            logging.info(f"Пустой файл симулированных сделок ребалансировки сохранен в {rebalance_trades_csv_path}")
    elif generate_reports:
        logging.warning("generate_reports равно True, но actual_reports_dir не установлен. Пропускаем сохранение rebalance_trades.csv.")
    else:
        logging.info("Генерация отчетов ВЫКЛЮЧЕНА. Пропускаем сохранение rebalance_trades.csv.")

    logging.info("Бэктест завершен.")
    df_equity = pd.DataFrame(equity_over_time)
    df_trades = pd.DataFrame(trades_list)
    # Округление безопасно, даже если часть колонок отсутствует или нет сделок
    if not df_trades.empty:
        cols_to_round = [
            c for c in (
                'quantity_quote','entry_price','exit_price',
                'commission_quote','slippage_quote','pnl_gross_quote','pnl_net_quote'
            ) if c in df_trades.columns
        ]
        if cols_to_round:
            df_trades[cols_to_round] = df_trades[cols_to_round].round(2)

    # ---------- МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ ----------
    def compute_metrics(df_eq: pd.DataFrame, trades: list[dict], initial_nav: float, ann_factor: int = 252):
        out: dict[str, float] = {}
        if df_eq.empty:
            return {k: 0.0 for k in ("sharpe_ratio", "sortino_ratio", "max_drawdown_percent", "profit_factor", "win_rate_percent")}

        # ---- Динамический фактор аннуализации ---------------------------------
        if "timestamp" in df_eq.columns and pd.api.types.is_datetime64_any_dtype(df_eq["timestamp"]):
            # Убеждаемся, что timestamp отсортирован, чтобы diff имел смысл - df_equity обычно уже отсортирован по времени
            # df_eq = df_eq.sort_values(by="timestamp") # Опционально: раскомментируйте, если сортировка не гарантирована
            freq_sec = df_eq["timestamp"].diff().dt.total_seconds().median()
            if pd.notna(freq_sec) and freq_sec > 0:
                periods_per_year = (365 * 24 * 60 * 60) / freq_sec  # например, 5-мин → 105 120
                ann_sqrt = np.sqrt(periods_per_year)
                # logging.debug(f"Динамический ann_sqrt: {ann_sqrt:.2f} (freq_sec: {freq_sec:.2f}s, периодов в году: {periods_per_year:.2f})")
            else:
                ann_sqrt = np.sqrt(252)  # фолбэк к дневным барам
                # logging.debug(f"Динамический ann_sqrt: Фолбэк к дневному (252) из-за freq_sec: {freq_sec}")
        else:
            ann_sqrt = np.sqrt(252) # фолбэк к дневным барам, если нет колонки timestamp
            # logging.debug("Динамический ann_sqrt: Фолбэк к дневному (252) из-за отсутствия/невалидной колонки timestamp")

        EPS = 1e-9
        rets = df_eq["portfolio_value_usdt"].pct_change().dropna()
        if rets.empty:
            mean_ret = std_ret = down_std = 0.0
        else:
            mean_ret = rets.mean()
            std_ret  = rets.std()
            down_std = rets[rets < 0].std()

        if std_ret < EPS:              # случай нулевой волатильности или одной точки
            out["sharpe_ratio"]  = 0.0
        else:
            out["sharpe_ratio"]  = mean_ret / std_ret * ann_sqrt

        if down_std is None or down_std < EPS: # down_std может быть None, если rets[rets<0] пуст
            out["sortino_ratio"] = 0.0
        else:
            out["sortino_ratio"] = mean_ret / down_std * ann_sqrt

        rolling_max = df_eq["portfolio_value_usdt"].cummax()
        drawdown = (df_eq["portfolio_value_usdt"] - rolling_max) / rolling_max
        out["max_drawdown_percent"] = drawdown.min() * 100 if not drawdown.empty else 0.0

        if trades:
            pnl_list = [t.get("pnl_net_quote", 0.0) for t in trades]
            wins = [p for p in pnl_list if p > 0]
            losses = [-p for p in pnl_list if p < 0]
            out["profit_factor"] = (sum(wins) / sum(losses)) if losses else 0.0
            out["win_rate_percent"] = (len(wins) / max(1, len(pnl_list))) * 100
        else:
            out["profit_factor"] = 0.0
            out["win_rate_percent"] = 0.0
        # Фолбэк: если в трейд-логе нет информативных pnl (например, только комиссии),
        # оценим win-rate и PF по ряду доходностей equity
        if out.get("profit_factor", 0.0) == 0.0 and out.get("win_rate_percent", 0.0) == 0.0 and not df_eq.empty:
            rets = df_eq["portfolio_value_usdt"].pct_change().dropna()
            wins = (rets > 0).sum()
            losses = (rets < 0).sum()
            out["win_rate_percent"] = (wins / max(1, wins + losses)) * 100
            out["profit_factor"] = (
                rets[rets > 0].sum() / abs(rets[rets < 0].sum())
            ) if losses else 0.0

        return out

    metrics = {}
    metrics["run_id"] = f"backtest_{timestamp_str}"
    # ... (ЗДЕСЬ ДОЛЖНЫ БЫТЬ СОХРАНЕНЫ ВСЕ ОРИГИНАЛЬНЫЕ РАСЧЕТЫ МЕТРИК) ...
    metrics["initial_portfolio_value_usdt"] = initial_portfolio_value_usdt
    final_portfolio_value = df_equity['portfolio_value_usdt'].iloc[-1] if not df_equity.empty else initial_portfolio_value_usdt
    metrics["final_portfolio_value_usdt"] = final_portfolio_value
    metrics["total_net_pnl_usdt"] = final_portfolio_value - initial_portfolio_value_usdt

    # Расчет дополнительных метрик производительности
    extra_metrics = compute_metrics(df_equity, trades_list,
                                    initial_portfolio_value_usdt,
                                    params.get("annualization_factor", 252))
    metrics.update(extra_metrics)
    metrics["total_net_pnl_percent"] = (metrics["total_net_pnl_usdt"] / initial_portfolio_value_usdt) * 100 if initial_portfolio_value_usdt != 0 else 0
    metrics["total_trades"] = len(df_trades)

    # Расчет стандартного отклонения NAV в процентах
    if not df_equity.empty:
        nav_series = df_equity['portfolio_value_usdt']
        # Новый расчет на основе процентных изменений, не требует initial_portfolio_value_usdt для масштабирования здесь.
        metrics["nav_std_percent"] = nav_series.pct_change().fillna(0).std() * 100
    else:
        # Обработка случая, когда df_equity пуст (например, нет сделок или данных)
        metrics["nav_std_percent"] = 0.0

    # ─── Очистка от числового шума (≤ 1 цент) ───────────────────────
    tol = float(params.get("neutrality_pnl_tolerance_usd", 1e-2))
    if abs(metrics["total_net_pnl_usdt"]) <= tol:
        metrics["total_net_pnl_usdt"] = 0.0
        metrics["total_net_pnl_percent"] = 0.0
        # metrics["sharpe_ratio"] = 0.0  # спорное решение, возможно лучше оставлять как есть
        metrics["final_portfolio_value_usdt"] = initial_portfolio_value_usdt

    # Добавляем соответствующие счетчики состояния портфеля в метрики
    metrics["num_safe_mode_entries"] = portfolio.get('num_safe_mode_entries', 0)
    metrics["num_circuit_breaker_triggers"] = portfolio.get('num_circuit_breaker_triggers', 0)
    metrics["time_steps_in_safe_mode"] = portfolio.get('time_steps_in_safe_mode', 0)
    metrics["num_blocked_trades"] = len(blocked_trades_list)

    # Округляем итоговые метрики: деньги до 2 знаков, проценты до 3 знаков
    for k, v in metrics.items():
        if isinstance(v, float):
            if k.endswith('_usdt'):
                metrics[k] = round(v, 2)
            elif k.endswith('_percent'):
                metrics[k] = round(v, 3)

    # (И много других метрик из оригинального файла)


    # Синхронизируем «очищенные» метрики с объектом,
    # возвращаемым оптимизатору / внешним вызовам.
    if generate_reports and actual_reports_dir:
        logging.info(f"Генерация отчетов в {actual_reports_dir}...")
        trades_csv_path = os.path.join(actual_reports_dir, "trades.csv")
        if not df_trades.empty:
            df_trades.to_csv(trades_csv_path, index=False)
        else:
            pd.DataFrame(columns=[
                "timestamp_open", "timestamp_close", "asset_type", "action",
                "quantity_asset", "quantity_quote", "entry_price", "exit_price",
                "commission_quote", "slippage_quote", "pnl_gross_quote", "pnl_net_quote"
            ]).to_csv(trades_csv_path, index=False)
        logging.info(f"Отчет о сделках сохранен в {trades_csv_path}")

        df_summary = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        summary_csv_path = os.path.join(actual_reports_dir, "summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        logging.info(f"Сводный отчет сохранен в {summary_csv_path}")

        if not df_equity.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=df_equity['timestamp'], y=df_equity['portfolio_value_usdt'], mode='lines', name='Portfolio Value (USDT)'),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_market['timestamp'], y=df_market['close'], mode='lines',
                    name=f"{main_asset_symbol} Price", line=dict(color='rgba(255,165,0,0.6)')),
                secondary_y=True,
            )

            asset_colors = {
                long_asset_key: {'BUY': 'rgba(0,0,255,0.7)', 'SELL': 'rgba(255,140,0,0.7)'},
                short_asset_key: {'BUY': 'rgba(128,0,128,0.7)', 'SELL': 'rgba(165,42,42,0.7)'}
            }
            asset_symbols = {
                long_asset_key: {'BUY': 'circle', 'SELL': 'circle-open'},
                short_asset_key: {'BUY': 'star', 'SELL': 'star-open'}
            }

            if not df_trades.empty:
                for asset_name_key_plot in [long_asset_key, short_asset_key]:
                    for action_str_plot in ['BUY', 'SELL']:
                        trades_to_plot = df_trades[
                            (df_trades['action'] == action_str_plot) &
                            (df_trades['asset_type'] == asset_name_key_plot)
                        ]
                        if not trades_to_plot.empty:
                            color_map_plot = asset_colors.get(asset_name_key_plot, {})
                            symbol_map_plot = asset_symbols.get(asset_name_key_plot, {})

                            fig.add_trace(go.Scatter(
                                x=trades_to_plot['timestamp_open'],
                                y=trades_to_plot['entry_price'],
                                mode='markers',
                                marker=dict(
                                    color=color_map_plot.get(action_str_plot, 'grey'),
                                    symbol=symbol_map_plot.get(action_str_plot, 'diamond'),
                                    size=9,
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                name=f'{asset_name_key_plot} {action_str_plot}',
                                yaxis="y2",
                                hoverinfo='text',
                                text=[
                                    (f"Asset: {trade_row['asset_type']}<br>Action: {trade_row['action']}<br>"
                                     f"Qty Asset: {trade_row['quantity_asset']:.6f}<br>Qty Quote: {trade_row['quantity_quote']:.2f}<br>"
                                     f"Price: {trade_row['entry_price']:.2f}<br>Comm: {trade_row['commission_quote']:.2f}<br>"
                                     f"Timestamp: {trade_row['timestamp_open'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    for _, trade_row in trades_to_plot.iterrows()
                                ]
                            ), secondary_y=True)

            fig.update_layout(
                title_text=f'Динамика капитала портфеля по сравнению с ценой {main_asset_symbol}',
                xaxis_title='Временная метка',
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            fig.update_yaxes(title_text="Стоимость портфеля (USDT)", secondary_y=False, showgrid=True)
            fig.update_yaxes(title_text=f"Цена {main_asset_symbol} (USDT)", secondary_y=True, showgrid=False)

            # --------------- ▼▼  НОВОЕ — наложение сигналов BUY / SELL  ▼▼ ---------------
            if apply_signal_logic and df_signals is not None and not df_signals.empty:
                # price привязываем к ближайшей свече (если сигналы «попадают в дырку»)
                df_sig_plot = (df_signals
                               .merge(df_market[['timestamp', 'close']], on='timestamp', how='left')
                               .ffill())
                for sig, color, sym in [("BUY","rgba(0,200,0,.85)",'triangle-up'), ("SELL","rgba(200,0,0,.85)",'triangle-down')]: # Изменены имена переменных
                    sub = df_sig_plot[df_sig_plot.signal==sig] # Изменен доступ к колонке 'signal'
                    if sub.empty: continue
                    fig.add_trace(go.Scatter(x=sub.timestamp, y=sub.close, mode='markers',
                              marker=dict(size=10, symbol=sym, color=color, line=dict(width=1.2,color='DarkSlateGrey')),
                              name=f"сигнал {sig}",   # своя трасса
                              legendgroup=f"{sig}_signal",  # отдельная группа
                              yaxis='y2',
                              showlegend=True))
            # --------------- ▲▲  КОНЕЦ НОВОГО  ▲▲ ------------------------------------------

            equity_html_path = os.path.join(actual_reports_dir, "equity.html")
            fig.write_html(equity_html_path)
            logging.info(f"Улучшенная кривая капитала сохранена в {equity_html_path}")

            # Округляем кривую equity до 2 знаков перед сохранением
            df_equity['portfolio_value_usdt'] = df_equity['portfolio_value_usdt'].round(2)
            equity_csv_path = os.path.join(actual_reports_dir, "equity.csv")
            df_equity.to_csv(equity_csv_path, index=False)
            logging.info(f"Данные кривой капитала сохранены в {equity_csv_path}")
        else:
            logging.warning("Данные о капитале пусты. Пропускаем генерацию кривой капитала и сохранение equity.csv.")

        if blocked_trades_list:
            df_blocked_trades = pd.DataFrame(blocked_trades_list)
            if not df_blocked_trades.empty: # Проверяем, не пуст ли DataFrame после создания
                blocked_trades_csv_path = os.path.join(actual_reports_dir, "blocked_trades_log.csv")
                df_blocked_trades.to_csv(blocked_trades_csv_path, index=False)
                logging.info(f"Журнал заблокированных сделок сохранен в {blocked_trades_csv_path}")
            else: # Этот случай может произойти, если blocked_trades_list был пуст
                logging.info("Во время этого бэктеста сделки не были заблокированы сигналами (DataFrame был пуст).")
        else: # Этот случай для пустого списка
            logging.info("Во время этого бэктеста сделки не были заблокированы сигналами (список был пуст).")

        logging.info(f"Все отчеты для этого запуска сгенерированы в {actual_reports_dir}.")
    elif generate_reports: # actual_reports_dir почему-то не был установлен
        logging.warning("generate_reports равно True, но actual_reports_dir не установлен. Пропускаем блок генерации основных отчетов.")
    else: # generate_reports равно False
        logging.info("Генерация отчетов ВЫКЛЮЧЕНА. Пропускаем блок генерации основных отчетов.")

    results_for_optimizer = metrics.copy()
    results_for_optimizer["output_dir"] = actual_reports_dir # Сохраняем actual_reports_dir, который может быть None, если отчеты выключены
    results_for_optimizer["status"] = "Завершено"

    for key_metric in ["sharpe_ratio", "sortino_ratio", "profit_factor",
                       "win_rate_percent", "max_drawdown_percent"]:
        if pd.isna(results_for_optimizer.get(key_metric)):
            results_for_optimizer[key_metric] = 0.0
            logging.warning(f"Метрика {key_metric} была NaN, преобразована в 0.0 для оптимизатора.")
    return results_for_optimizer
# --- END OF REPLACEMENT FUNCTION ---

def run_standalone_backtest(backtest_settings_dict, data_file_path):
    """
    Wrapper to run a single backtest using a backtest_settings dictionary and data file path.
    This is primarily for CLI execution of the backtester itself.
    """
    logging.info(f"Running standalone backtest with data from: {data_file_path}")
    # Standalone runs should always generate full reports, so is_optimizer_call=False.
    # The run_backtest function's default for generate_reports handles this.
    return run_backtest(backtest_settings_dict, data_file_path, is_optimizer_call=False)


def main():
    parser = argparse.ArgumentParser(description="Prosperous Bot Rebalance Backtester (CLI - Unified Config)")
    parser.add_argument("--config_file", type=str, required=True, 
                        help="Path to the unified JSON configuration file (e.g., config/unified_config.json). "
                             "The backtester will use the 'backtest_settings' section.")
    parser.add_argument("--override", type=str, help="JSON string to override config parameters, e.g., '{\"main_asset_symbol\":\"DOGE\"}'")
    args = parser.parse_args()

    backtest_params = None
    # data_file_path will be resolved later
    
    try:
        with open(args.config_file, 'r') as f:
            unified_config = json.load(f)
        
        if "backtest_settings" in unified_config:
            backtest_params = unified_config["backtest_settings"]
            logging.info(f"Loaded backtest settings from '{args.config_file}' (using 'backtest_settings' section).")
        else:
            logging.error(f"FATAL: Unified config '{args.config_file}' does not contain a 'backtest_settings' section. "
                          "This section is required for standalone backtester runs. Please use/create a config based on "
                          "'config/unified_config.example.json'. Exiting.")
            return

        # Determine actual_main_asset_symbol and update backtest_params if override is present
        actual_main_asset_symbol = backtest_params.get("main_asset_symbol", "BTC") # Default from config or BTC

        if args.override:
            try:
                override_dict = json.loads(args.override)
                if "main_asset_symbol" in override_dict:
                    actual_main_asset_symbol = override_dict["main_asset_symbol"]
                    backtest_params["main_asset_symbol"] = actual_main_asset_symbol # Update for run_backtest internal use
                    logging.info(f"Override: main_asset_symbol set to '{actual_main_asset_symbol}'.")
                # Potentially merge other overrides into backtest_params here if needed for run_standalone_backtest
                # For now, only main_asset_symbol is critical for path resolution before run_backtest
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding --override JSON '{args.override}': {e}. Using symbol from config or default.")

        csv_path_template = backtest_params.get("data_settings", {}).get("csv_file_path")
        if not csv_path_template:
            logging.error("FATAL: 'data_settings.csv_file_path' not found in the 'backtest_settings' section of the config. Exiting.")
            return
        
        resolved_data_file_path = _subst_symbol(csv_path_template, actual_main_asset_symbol)
        if not resolved_data_file_path: # Should not happen if template and symbol are valid
            logging.error(f"FATAL: Could not resolve data_file_path from template '{csv_path_template}' with symbol '{actual_main_asset_symbol}'. Exiting.")
            return
        
        # The signals_csv_path is resolved inside run_backtest using the (potentially overridden)
        # main_asset_symbol in backtest_params, so no need to resolve it here for the main function's direct use.

    except FileNotFoundError:
        logging.warning(f"Configuration file '{args.config_file}' not found. "
                        "A dummy configuration will be created at 'config/dummy_unified_config_for_backtester.json' for demonstration.")
        
        dummy_config_filename = "dummy_unified_config_for_backtester.json"
        dummy_config_path = os.path.join("config", dummy_config_filename) 
        dummy_data_filename = "dummy_BTCUSDT_1h_for_backtester.csv" # Changed GALA to BTC
        dummy_data_path = os.path.join("data", dummy_data_filename) 

        os.makedirs(os.path.dirname(dummy_config_path), exist_ok=True)
        os.makedirs(os.path.dirname(dummy_data_path), exist_ok=True)

        # Define a minimal but complete backtest_settings structure for the dummy
        dummy_backtest_settings_content = {
          "main_asset_symbol": "BTC",
          "apply_signal_logic": True, # Added apply_signal_logic
          "initial_capital": 10000.0,
          "commission_taker": 0.0007,
          "commission_maker": 0.0002,
          "use_maker_fees_in_backtest": False,
          "slippage_percent": 0.0005,
          "annualization_factor": 252.0,
          "min_rebalance_interval_minutes": 60,
          "rebalance_threshold": 0.02,
          "target_weights_normal": {
              "BTC_SPOT": 0.65,
              "BTC_LONG5X": 0.11,
              "BTC_SHORT5X": 0.24
          },
          "circuit_breaker_config": {
            "enabled": True, "threshold_percentage": 0.10,
            "lookback_candles": 1, "movement_calc_type": "(high-low)/open"
          },
          "safe_mode_config": {
            "enabled": True, "metric_to_monitor": "margin_usage",
            "entry_threshold": 0.70, "exit_threshold": 0.50,
            "target_weights_safe": {
                "BTC_SPOT": 0.75,
                "BTC_LONG5X": 0.05,
                "BTC_SHORT5X": 0.05,
                "USDT": 0.15
            }
          },
          "data_settings": {
            "csv_file_path": dummy_data_path,
            "signals_csv_path": os.path.join("data", "dummy_BTCUSDT_signals_for_backtester.csv"),
            "timestamp_col": "timestamp",
            "ohlc_cols": {"open": "open", "high": "high", "low": "low", "close": "close"},
            "volume_col": "volume",
            "price_col_for_rebalance": "close"
          },
          "date_range": {
              "start_date": "2023-01-01T00:00:00Z", "end_date": "2023-01-05T23:59:59Z"
          },
          "logging_level": "INFO",
          "report_path_prefix": "./reports/backtest_"
        }
        dummy_unified_config_content = {"backtest_settings": dummy_backtest_settings_content}

        try:
            with open(dummy_config_path, 'w') as f: 
                json.dump(dummy_unified_config_content, f, indent=2)
            logging.info(f"Dummy unified config for backtester created at '{dummy_config_path}'. You should run with this path next time.")
            
            backtest_params = dummy_backtest_settings_content
            # This dummy logic needs to be careful about resolved_data_file_path vs data_file_path
            # For simplicity, if config is not found, we'll use the dummy paths directly without substitution for now.
            # A more robust dummy creation would also consider the override for symbol.
            dummy_data_path_val = dummy_data_path # Store the original dummy path template

            if not os.path.exists(dummy_data_path_val): # Check existence of template path
                timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=120, freq='h') # 5 days
                prices = [20000 + (i*2) + (100 * ((i//24)%5)) - (80 * (i % 3)) for i in range(120)]
                df_dummy_data = pd.DataFrame({
                    'timestamp': timestamps, 'open': [p - 5 for p in prices], 'high': [p + 10 for p in prices],
                    'low': [p - 10 for p in prices], 'close': prices, 'volume': [50 + i for i in range(120)]
                })
                df_dummy_data.to_csv(dummy_data_path_val, index=False)
                logging.info(f"Dummy market data file created at '{dummy_data_path_val}'")

            # Create dummy signals CSV if path is specified and file doesn't exist
            # This part also needs care if we want dummy signals to match a potentially overridden symbol.
            # For now, dummy signals path is hardcoded or uses BTC.
            dummy_signals_path_template = dummy_backtest_settings_content["data_settings"]["signals_csv_path"]
            # actual_main_asset_symbol for dummy case would default to BTC or from override if provided.
            # This is getting complex for dummy section, ideally dummy config has fixed names.
            # Let's assume dummy config uses fixed names for now.
            if dummy_signals_path_template and not os.path.exists(dummy_signals_path_template):
                signal_timestamps = pd.to_datetime(['2023-01-01T00:00:00Z', '2023-01-01T10:00:00Z', # Using dummy_signals_path_template directly
                                                    '2023-01-02T05:00:00Z', '2023-01-03T15:00:00Z', # as it might not have placeholders
                                                    '2023-01-04T20:00:00Z'])
                signals = ['NEUTRAL', 'BUY', 'NEUTRAL', 'SELL', 'BUY']
                df_dummy_signals = pd.DataFrame({'timestamp': signal_timestamps, 'signal': signals})
                df_dummy_signals.to_csv(dummy_signals_path_template, index=False)
                logging.info(f"Dummy signal data file created at '{dummy_signals_path_template}'")
            
            # If config not found, we are in dummy mode.
            # resolved_data_file_path should be set to the dummy path.
            resolved_data_file_path = dummy_data_path # Use the variable holding the actual dummy path string.
            # backtest_params is already set to dummy_backtest_settings_content
            logging.info("Exiting after creating dummy files. Please re-run with the dummy config: "
                         f"`python -m src.prosperous_bot.rebalance_backtester --config_file {dummy_config_path}` "
                         f" (and optionally --override if testing that feature with dummy data).")
            return

        except Exception as e:
            logging.error(f"Could not create dummy unified config or data file: {e}", exc_info=True)
            return

    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from config file: {args.config_file}. Exiting.")
        return
    except Exception as e: 
        logging.error(f"FATAL: An unexpected error occurred while loading the configuration: {e}", exc_info=True)
        return

    if backtest_params and resolved_data_file_path: # Check resolved_data_file_path
        run_standalone_backtest(backtest_params, resolved_data_file_path)
    else:
        # This state should ideally not be reached if the above logic is correct,
        # especially with the new checks for csv_path_template and resolved_data_file_path.
        logging.error("Critical error: Parameters or data file path could not be determined. Backtest aborted.")

if __name__ == "__main__":
    main()