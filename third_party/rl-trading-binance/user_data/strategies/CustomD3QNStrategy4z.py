import sys
import os
import logging
import logging.handlers
import importlib.util
from pathlib import Path
import json
import numpy as np  # type: ignore
import pandas as pd
from pandas import DataFrame
import torch  # type: ignore
from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
import onnxruntime as ort
import threading
from collections import OrderedDict
from typing import Dict, Optional, List, Any, Tuple
from collections import deque
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


try:
    from freqtrade.persistence import Trade  # type: ignore
except ImportError:
    class Trade:
        id: int
        pair: str
        is_open: bool
        is_short: bool
        open_date_utc: Optional[datetime] = None
        close_rate_requested: Optional[float] = None
        open_rate: float = 0.0
        close_profit: Optional[float] = None
        
        def calc_profit(self, rate: float) -> float: return 0.0

        @classmethod
        def get_trades(cls, filters): return []

        @classmethod
        def get_open_trades(cls): return []

from datetime import datetime, timezone, timedelta

# --- 1. НАСТРОЙКА ПУТЕЙ ---
strategy_file = Path(__file__).resolve()
if strategy_file.parent.name == 'strategies':
    project_root = strategy_file.parent.parent.parent
else:
    project_root = strategy_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Freqtrade imports
try:
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter, merge_informative_pair  # type: ignore
except ImportError:
    logging.getLogger(__name__).error("Could not import freqtrade.strategy")
    class IStrategy:
        dp: Any = None
        config: Dict[str, Any]
        logger: Any = None
        def __init__(self, config: dict):
            self.config = config
            self.logger = logging.getLogger(__name__)
    class DecimalParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0.0)
    class IntParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0)
    class CategoricalParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', args[0][0] if args and args[0] else None)
    # Fallback для merge_informative_pair, чтобы не ломать оффлайн-инструменты
    def merge_informative_pair(dataframe, informative, timeframe, informative_timeframe, ffill=True):
        return dataframe

logger = logging.getLogger(__name__)

# Agent imports
try:
    from agent import D3QN_PER_Agent  # type: ignore
except ImportError as e:
    logger.error(f"CRITICAL: Could not import D3QN Agent! Check path: {project_root}")
    raise e


class CustomD3QNStrategy4z(IStrategy):
    config: Dict[str, Any]
    dp: Any
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short: bool = True  # Это критично для Futures режима
    startup_candle_count: int = 180

    # Информативный таймфрейм для режима рынка
    informative_timeframe = '1m'
    informative_timeframe_global = '1m'
    
    # minimal_roi = {"0": 100}
    minimal_roi = {
        "59": 0.001
    }
    stoploss = -0.21
    trailing_stop = False
    use_custom_stoploss = True

    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Дисконт для Maker-ордеров (от 0% до 1%)
    entry_discount_pct = DecimalParameter(0.0, 0.01, default=0.004, space='buy', optimize=False, load=False)
    
    # Параметры TSL (КОНСЕРВАТИВНЫЕ) d0 0.547, d_min 0.001, hysteresis 0.002, p_target 0.015, tsl_exponent 0.953
    d0 = DecimalParameter(0.01, 1.0, default=0.547, space='sell', optimize=False, load=False)
    d_min = DecimalParameter(0.0005, 0.005, default=0.001, space='sell', optimize=False, load=False)
    hysteresis = DecimalParameter(0.001, 0.005, default=0.002, space='sell', optimize=False, load=False)
    p_target = DecimalParameter(0.005, 0.05, default=0.015, space='sell', optimize=False, load=False)
    
    # Степень нелинейности TSL (1.0 - Линейно для предсказуемости)
    tsl_exponent = DecimalParameter(0.9, 1.05, default=0.953, space='sell', optimize=False, load=False)

    # Hyperoptable Voting Thresholds (СТРОГО 2 из 2)
    rl_long_threshold_opt = IntParameter(2, 2, default=2, space='buy', optimize=False, load=False)
    rl_short_threshold_opt = IntParameter(2, 2, default=2, space='sell', optimize=False, load=False)

    # Пороги для ВЫХОДА (Alpha Decay) - 1 (любая модель) или 2 (консенсус)
    rl_exit_long_threshold = IntParameter(1, 2, default=2, space='sell', optimize=False, load=False)
    rl_exit_short_threshold = IntParameter(1, 2, default=2, space='sell', optimize=False, load=False)

    # Оптимизируемый таймфрейм для глобального режима (теперь локальный старший ТФ)
    global_ema_timeframe = CategoricalParameter(['1m', '3m', '5m', '15m', '30m', '1h'], default='1m', space='buy', optimize=False, load=False)

    # EMA фильтр (быстрый, для локального режима)
    ema_fast_period = IntParameter(10, 15, default=12, space='buy', optimize=True, load=False)

    # EMA фильтр (на старшем ТФ)
    global_ema_period = IntParameter(5, 60, default=25, space='buy', optimize=True, load=False)

    # Фильтр по объему (Фокус на ликвидности)
    min_quote_volume_usd = DecimalParameter(0, 500000, default=100000, space='buy', optimize=False, load=False)

    # Коэффициент агрессии для Dynamic Epsilon
    dd_aggression_k = DecimalParameter(0.0, 2.0, default=2.0, space='buy', optimize=False, load=False)

    # Оптимизируемые пороги уверенности (Epsilon) - ВЫСОКИЙ ПОРОГ rl_epsilon_long 0.209 rl_epsilon_short 0.743
    rl_epsilon_long = DecimalParameter(0.0001, 0.01, default=0.001, space='buy', optimize=False, load=False)
    rl_epsilon_short = DecimalParameter(0.0001, 0.01, default=0.001, space='sell', optimize=False, load=False)

    # --- DYNAMIC VOLUME WINDOWS ---
    vol_window = IntParameter(28, 34, default=31, space='buy', optimize=True, load=False)
    cvd_window = IntParameter(75, 90, default=83, space='buy', optimize=True, load=False)

    # --- TOGGLES FOR VOLUME FILTERS (Enable/Disable individually) ---
    vol_f1_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=False)
    vol_f2_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=False)
    vol_f3_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=False)

    # --- EXPANDED VOLUME FILTERS (Wider Ranges for Early Entry) --- # Мы начинаем поиск с 1.1 (чуть выше нормы), чтобы поймать импульс в зародыше
    vol_f1_surge = DecimalParameter(1.05, 3.0, default=2.627, space='buy', optimize=False, load=False)
    vol_f1_pct = DecimalParameter(51.0, 80.0, default=79.172, space='buy', optimize=False, load=False)

    vol_f2_cvd_spike = DecimalParameter(1.0, 2.5, default=1.024, space='buy', optimize=True, load=False)
    vol_f2_gap = DecimalParameter(1.1, 2.5, default=1.959, space='buy', optimize=True, load=False)

    # --- EXIT CLIMAX (More aggressive) ---
    vol_f3_peak = DecimalParameter(4.5, 6.0, default=5.857, space='sell', optimize=False, load=False)
    vol_f3_fade = DecimalParameter(0.2, 0.3, default=0.276, space='sell', optimize=False, load=False)

    # Экстренный выход по сигналу ансамбля при достижении порога убытка (Alpha Stop) (Для проверки в Live поставьте -0.001)
    emergency_exit_threshold = DecimalParameter(-0.3, 0.0, default=-0.99, space='sell', optimize=True, load=False)

    # --- ATR DYNAMIC FLOOR PARAMETERS ---
    atr_multiplier = DecimalParameter(1.5, 5.0, default=1.572, space='sell', optimize=False, load=False)
    atr_period = IntParameter(10, 40, default=24, space='sell', optimize=False, load=False)


    plot_config = {
        'main_plot': {},
        'subplots': {
            # Визуальный контроль режимов
            'regimes': {
                'st_regime_global': {'color': 'orange'},
                'st_regime_local': {'color': 'blue'},
            },
            'volume': {
                'quote_volume_sma': {'color': 'green', 'type': 'line'},
            }
        },
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)  # type: ignore
        
        # --- STAGE 3: SECURITY (Environment Variables) ---
        if load_dotenv:
            env_path = project_root / '.env'
            load_dotenv(dotenv_path=env_path)

        # Override sensitive data from environment variables
        if os.environ.get('FT_PASSWORD'):
            self.config.get('api_server', {})['password'] = os.environ.get('FT_PASSWORD')
        if os.environ.get('FT_JWT_SECRET'):
            self.config.get('api_server', {})['jwt_secret_key'] = os.environ.get('FT_JWT_SECRET')
        if os.environ.get('EXCHANGE_KEY'):
            self.config.get('exchange', {})['key'] = os.environ.get('EXCHANGE_KEY')
        if os.environ.get('EXCHANGE_SECRET'):
            self.config.get('exchange', {})['secret'] = os.environ.get('EXCHANGE_SECRET')

        # --- 1. ПАРАМЕТРЫ ПРАВИЛ ТОРГОВЛИ ---
        if 'stoploss' in config:
            self.stoploss = config['stoploss']
        if 'minimal_roi' in config:
            self.minimal_roi = config['minimal_roi']
        if 'trailing_stop' in config:
            self.trailing_stop = config['trailing_stop']
        if 'use_custom_stoploss' in config:
            self.use_custom_stoploss = config['use_custom_stoploss']

        # --- 2. ПАРАМЕТРЫ TSL (ОБНОВЛЕНИЕ ЧЕРЕЗ КОНФИГ) ---
        rl_tsl = config.get('rl_tsl', {})
        if 'd0' in rl_tsl: self.d0.value = float(rl_tsl['d0'])
        if 'd_min' in rl_tsl: self.d_min.value = float(rl_tsl['d_min'])
        if 'hysteresis' in rl_tsl: self.hysteresis.value = float(rl_tsl['hysteresis'])
        if 'p_target' in rl_tsl: self.p_target.value = float(rl_tsl['p_target'])
        if 'tsl_exponent' in rl_tsl: self.tsl_exponent.value = float(rl_tsl['tsl_exponent'])

        # --- 3. ГОЛОСОВАНИЕ И ПОРОГИ (VOTING) ---
        if 'rl_long_threshold' in config: self.rl_long_threshold_opt.value = int(config['rl_long_threshold'])
        if 'rl_short_threshold' in config: self.rl_short_threshold_opt.value = int(config['rl_short_threshold'])
        
        # Epsilon (Thresholds)
        rl_ens = config.get('rl_ensemble', {})
        if 'epsilon_threshold_long' in rl_ens: self.rl_epsilon_long.value = float(rl_ens['epsilon_threshold_long'])
        if 'epsilon_threshold_short' in rl_ens: self.rl_epsilon_short.value = float(rl_ens['epsilon_threshold_short'])

        # --- 4. РЕЖИМЫ И EMA (REGIME) ---
        rl_regime = config.get('rl_regime', {})
        if 'global_ema_timeframe' in rl_regime: self.global_ema_timeframe.value = str(rl_regime['global_ema_timeframe'])
        if 'ema_fast_period' in rl_regime: self.ema_fast_period.value = int(rl_regime['ema_fast_period'])
        if 'global_ema_period' in rl_regime: self.global_ema_period.value = int(rl_regime['global_ema_period'])

        if hasattr(self, 'global_ema_timeframe'):
            self.informative_timeframe_global = self.global_ema_timeframe.value
            
        # Принудительно включаем шорты
        self.can_short = config.get('can_short', True)
        
        # --- PERFORMANCE OPTIMIZATION: GLOBAL REGIME CACHE ---
        self._global_regime_cache = {}
        self._global_regime_lock = threading.Lock()
        
        # --- 5. ФИЛЬТРЫ ОБЪЕМА (VOLUME) ---
        if 'min_quote_volume_usd' in config:
            self.min_quote_volume_usd.value = float(config['min_quote_volume_usd'])
        
        if 'vol_f1_enabled' in rl_ens: self.vol_f1_enabled.value = bool(rl_ens['vol_f1_enabled'])
        if 'vol_f2_enabled' in rl_ens: self.vol_f2_enabled.value = bool(rl_ens['vol_f2_enabled'])
        if 'vol_f3_enabled' in rl_ens: self.vol_f3_enabled.value = bool(rl_ens['vol_f3_enabled'])
            
        if 'vol_window' in rl_ens: self.vol_window.value = int(rl_ens['vol_window'])
        if 'cvd_window' in rl_ens: self.cvd_window.value = int(rl_ens['cvd_window'])
            
        if 'vol_f1_surge' in rl_ens: self.vol_f1_surge.value = float(rl_ens['vol_f1_surge'])
        if 'vol_f1_pct' in rl_ens: self.vol_f1_pct.value = float(rl_ens['vol_f1_pct'])
        if 'vol_f2_cvd_spike' in rl_ens: self.vol_f2_cvd_spike.value = float(rl_ens['vol_f2_cvd_spike'])
        if 'vol_f2_gap' in rl_ens: self.vol_f2_gap.value = float(rl_ens['vol_f2_gap'])
        if 'vol_f3_peak' in rl_ens: self.vol_f3_peak.value = float(rl_ens['vol_f3_peak'])
        if 'vol_f3_fade' in rl_ens: self.vol_f3_fade.value = float(rl_ens['vol_f3_fade'])

        # Dynamic Epsilon Aggression
        if 'dd_aggression_k' in rl_ens: self.dd_aggression_k.value = float(rl_ens['dd_aggression_k'])

        # Экстренный выход (Alpha Stop) из конфига
        if 'emergency_exit_threshold' in config:
            self.emergency_exit_threshold.value = float(config['emergency_exit_threshold'])

        # --- 6. ENTRY DISCOUNT ---
        if 'entry_discount_pct' in config:
            self.entry_discount_pct.value = float(config['entry_discount_pct'])

        # --- LOGGING STATUS ---
        logger.info(f"[CONFIG] Volume Filter F1 (Surge): {'ENABLED' if self.vol_f1_enabled.value else 'DISABLED'}")
        logger.info(f"[CONFIG] Volume Filter F2 (CVD): {'ENABLED' if self.vol_f2_enabled.value else 'DISABLED'}")
        logger.info(f"[CONFIG] Volume Filter F3 (Exit): {'ENABLED' if self.vol_f3_enabled.value else 'DISABLED'}")
        logger.info(f"[CONFIG] TSL: d0={self.d0.value}, d_min={self.d_min.value}, exp={self.tsl_exponent.value}")
        logger.info(f"[CONFIG] Voting: L_thresh={self.rl_long_threshold_opt.value}, S_thresh={self.rl_short_threshold_opt.value}")
        logger.info(f"[CONFIG] Regime: Global TF={self.informative_timeframe_global}, Global Period={self.global_ema_period.value}")
        logger.info(f"[CONFIG] Entry Discount: {self.entry_discount_pct.value:.4%}")

        # Убираем спам о отмене стоплосса
        def filter_stoploss_cancel(record):
            msg = record.getMessage()
            return "Cancelling stoploss on exchange" not in msg and "Cancelling current stoploss on exchange" not in msg
        logging.getLogger('freqtrade.freqtradebot').addFilter(filter_stoploss_cancel)

        # Убираем спам о загрузке данных (Loading data for ... / data starts at ...)
        def filter_data_loading_spam(record):
            msg = record.getMessage()
            return "Loading data for" not in msg and "data starts at" not in msg
        logging.getLogger('freqtrade.data.dataprovider').addFilter(filter_data_loading_spam)
        logging.getLogger('freqtrade.data.history.datahandlers.idatahandler').addFilter(filter_data_loading_spam)
        
        # --- LOG ROTATION (DAILY) ---
        # Настраиваем ротацию логов раз в сутки (midnight), чтобы файл не рос бесконечно
        try:
            root_logger = logging.getLogger()
            handlers_to_swap = []
            
            for h in root_logger.handlers:
                # Ищем стандартный FileHandler (не ротируемый)
                if isinstance(h, logging.FileHandler) and not isinstance(h, logging.handlers.TimedRotatingFileHandler):
                    handlers_to_swap.append(h)
            
            for h in handlers_to_swap:
                # Создаем новый хендлер с ротацией
                new_handler = logging.handlers.TimedRotatingFileHandler(
                    filename=h.baseFilename,
                    when='midnight',
                    interval=1,
                    backupCount=1,  # Хранить архивы за 1 день
                    encoding='utf-8'
                )
                new_handler.setFormatter(h.formatter)
                new_handler.setLevel(h.level)
                
                root_logger.removeHandler(h)
                h.close()
                root_logger.addHandler(new_handler)
                logger.info(f"[LOG] Log rotation enabled for {h.baseFilename} (Daily at midnight)")
                
        except Exception as e:
            logger.warning(f"[WARNING] Log rotation setup failed: {e}")

        # === CPU ОПТИМИЗАЦИИ ===
        # 1. Установить количество потоков для PyTorch
        num_cpu_threads = config.get('cpu_threads', 4)  # по умолчанию 4 потока
        try:
            torch.set_num_threads(num_cpu_threads)
            torch.set_num_interop_threads(num_cpu_threads)
        except RuntimeError as e:
            logger.warning(f"[WARNING] Could not set torch threads (already initialized?): {e}")
        
        self.device = torch.device("cpu")
        
        self.logger = logging.getLogger(__name__)
        # 3. Кэш для feature tensors (экономим на preprocessing)
        self.feature_cache = OrderedDict()
        self.q_value_cache = OrderedDict()  # Кэш для результатов инференса (Q-values)
        self.cache_lock = threading.Lock()
        self.cache_max_size = 10  # Reduced from 100 to prevent OOM in backtesting/hyperopt
        
        if Path(__file__).parent.name == 'strategies':
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(__file__).parent.parent.parent
        
        self.tsl_memory = {}
        
        # История Advantage для автоподбора q_min/q_max
        self.adv_history = {}
        self.last_config_update = datetime.now()
        
        # Статистика конфликтов
        self.conflict_stats = {
            'total_signals': 0,
            'conflicts': 0,
            'long_entries': 0,
            'short_entries': 0,
        }

        # === DYNAMIC SLOT ALLOCATION (snake_case config: dynamic_slots) ===
        self.dynamic_slots_cfg = config.get('dynamic_slots', {})
        self.dynamic_slots_enabled = self.dynamic_slots_cfg.get('enabled', False)
        # max_open_trades comes from global config in snake_case
        self.total_slots = config.get('max_open_trades', 100)
        self.min_slots_per_side = self.dynamic_slots_cfg.get('min_slots_per_side', 10)
        self.aggression_factor = self.dynamic_slots_cfg.get('aggression_factor', 1.5)
        self.slot_update_interval = self.dynamic_slots_cfg.get('update_interval_sec', 300)

        self.max_long_slots = self.total_slots // 2 if self.total_slots > 0 else 50
        self.max_short_slots = self.total_slots - self.max_long_slots if self.total_slots > 0 else 50
        self.last_slot_update: Optional[datetime] = None
        self.slot_history = deque(maxlen=100)

        # --- LAZY-LOADED AGENTS & SESSIONS ---
        self._long_1_agent: Optional[D3QN_PER_Agent] = None
        self._long_2_agent: Optional[D3QN_PER_Agent] = None
        self._short_1_agent: Optional[D3QN_PER_Agent] = None
        self._short_2_agent: Optional[D3QN_PER_Agent] = None

        logger.info(
            f"🎰 Dynamic Slots: {'ENABLED' if self.dynamic_slots_enabled else 'DISABLED'} "
            f"| Total: {self.total_slots}"
        )

        # --- ПУТИ К 4 МОДЕЛЯМ (Load from Config) ---
        model_paths = config.get('rl_ensemble', {}).get('model_paths', {})
        self.long_1_model_dir = self.project_root / model_paths.get('long_1', 'default_long_1_path')
        self.long_1_model_pth = self.long_1_model_dir / "best.pth"

        self.long_2_model_dir = self.project_root / model_paths.get('long_2', 'default_long_2_path')
        self.long_2_model_pth = self.long_2_model_dir / "best.pth"

        self.short_1_model_dir = self.project_root / model_paths.get('short_1', 'default_short_1_path')
        self.short_1_model_pth = self.short_1_model_dir / "best.pth"

        self.short_2_model_dir = self.project_root / model_paths.get('short_2', 'default_short_2_path')
        self.short_2_model_pth = self.short_2_model_dir / "best.pth"
        
        # --- ВКЛЮЧЕНИЕ/ОТКЛЮЧЕНИЕ МОДЕЛЕЙ ---
        self.enable_long_1 = config.get('rl_enable_long_1', True)
        self.enable_long_2 = config.get('rl_enable_long_2', True)
        self.enable_short_1 = config.get('rl_enable_short_1', True)
        self.enable_short_2 = config.get('rl_enable_short_2', True)
        
        # --- ЗАГРУЗКА КОНФИГОВ ---
        logger.info("\n" + "="*60)
        logger.info("[START] INITIALIZING 2+2 ENSEMBLE SYSTEM")
        logger.info("="*60 + "\n")
        
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"[CONFIG] Active Models: L1={self.enable_long_1}, L2={self.enable_long_2}, S1={self.enable_short_1}, S2={self.enable_short_2}")
        
        # Long 1
        if self.enable_long_1:
            cfg_file_long_1 = self._find_config_file(self.long_1_model_dir)
            if not cfg_file_long_1:
                raise FileNotFoundError(f"Config not found in {self.long_1_model_dir}")
            logger.info(f"[OK] Loading LONG_1 config from {cfg_file_long_1}")
            self.cfg_long_1 = self._load_py_config(cfg_file_long_1)
        else:
            self.cfg_long_1 = None
        
        # Long 2
        if self.enable_long_2:
            cfg_file_long_2 = self._find_config_file(self.long_2_model_dir)
            if not cfg_file_long_2:
                raise FileNotFoundError(f"Config not found in {self.long_2_model_dir}")
            logger.info(f"[OK] Loading LONG_2 config from {cfg_file_long_2}")
            self.cfg_long_2 = self._load_py_config(cfg_file_long_2)
        else:
            self.cfg_long_2 = None
        
        # Short 1
        if self.enable_short_1:
            cfg_file_short_1 = self._find_config_file(self.short_1_model_dir)
            if not cfg_file_short_1:
                raise FileNotFoundError(f"Config not found in {self.short_1_model_dir}")
            logger.info(f"[OK] Loading SHORT_1 config from {cfg_file_short_1}")
            self.cfg_short_1 = self._load_py_config(cfg_file_short_1)
        else:
            self.cfg_short_1 = None
        
        # Short 2
        if self.enable_short_2:
            cfg_file_short_2 = self._find_config_file(self.short_2_model_dir)
            if not cfg_file_short_2:
                raise FileNotFoundError(f"Config not found in {self.short_2_model_dir}")
            logger.info(f"[OK] Loading SHORT_2 config from {cfg_file_short_2}")
            self.cfg_short_2 = self._load_py_config(cfg_file_short_2)
        else:
            self.cfg_short_2 = None
        
        # --- ОПРЕДЕЛЕНИЕ РЕЖИМА MIRROR MODE ---
        # Определяем из конфига модели
        self.short_1_is_mirror = getattr(self.cfg_short_1.market, 'mirror_mode', False) if self.cfg_short_1 is not None else False  # type: ignore
        self.short_2_is_mirror = getattr(self.cfg_short_2.market, 'mirror_mode', False) if self.cfg_short_2 is not None else False  # type: ignore
        
        logger.info(f"[INFO] SHORT_1 Mirror Mode: {self.short_1_is_mirror} (From Config)")
        logger.info(f"[INFO] SHORT_2 Mirror Mode: {self.short_2_is_mirror} (From Config)")
        
        # --- НАСТРОЙКИ АНСАМБЛЯ V2 (snake_case: rl_ensemble) ---
        self.ensemble_cfg = config.get('rl_ensemble', {})

        # Включение/выключение regime-фильтра (EMA на 1m/15m)
        self.use_global_regime_filter: bool = self.ensemble_cfg.get('use_global_regime_filter', True)
        self.use_local_regime_filter: bool = self.ensemble_cfg.get('use_local_regime_filter', True)

        # Base epsilon from config (common legacy value)
        self.epsilon_threshold: float = self.ensemble_cfg.get('epsilon_threshold', 0.15)

        # Separate base thresholds for long/short sides, defaulting to common epsilon
        self.epsilon_threshold_long: float = self.ensemble_cfg.get(
            "epsilon_threshold_long", self.epsilon_threshold
        )
        self.epsilon_threshold_short: float = self.ensemble_cfg.get(
            "epsilon_threshold_short", self.epsilon_threshold
        )

        # Effective epsilons used for thresholding, updated dynamically around per-side bases
        self.epsilon_threshold_eff_long: float = float(self.epsilon_threshold_long)
        self.epsilon_threshold_eff_short: float = float(self.epsilon_threshold_short)
        # Aggregate value for logging/compatibility
        self.epsilon_threshold_eff: float = 0.5 * (
            self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short
        )

        # Максимальная наблюдаемая equity по unrealized PnL для лонгов и шортов
        self.equity_max_long: float = 0.0
        self.equity_max_short: float = 0.0
        # Старое поле equity_max оставляем для обратной совместимости (не используется напрямую)
        self.equity_max: float = 0.0

        # RL voting and veto flags (snake_case)
        self.enable_veto: bool = config.get('rl_enable_veto', False)
        self.rl_long_threshold: int = config.get('rl_long_threshold', 1)
        self.rl_short_threshold: int = config.get('rl_short_threshold', 1)

        # --- Calibration Mode (snake_case: rl_calibration_mode) ---
        # When enabled, disable policy layers (veto, dynamic slots, dynamic epsilon)
        self.calibration_mode: bool = config.get("rl_calibration_mode", False)
        if self.calibration_mode:
            self.logger.warning(
                "⚠️ STRATEGY RUNNING IN CALIBRATION MODE! "
                "Veto, dynamic slots and dynamic epsilon are disabled."
            )
            # Disable veto and dynamic slots in calibration
            self.enable_veto = False
            self.dynamic_slots_enabled = False
            # Effective epsilons fixed to base values in calibration mode
            self.epsilon_threshold_eff_long = float(self.epsilon_threshold_long)
            self.epsilon_threshold_eff_short = float(self.epsilon_threshold_short)
            self.epsilon_threshold_eff = 0.5 * (
                self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short
            )
            
        # --- RUNMODE & CALIBRATION LOGGING ---
        self.runmode = config.get('runmode', 'unknown')
        self.logger.info(f"⚙️ Runmode: {self.runmode}")
        if self.calibration_mode:
            self.logger.info("⚠️ CALIBRATION MODE: Dynamic Epsilon & Veto DISABLED (Fixed to base values)")
        elif self.runmode in ('live', 'dry_run'):
            self.logger.info("[OK] LIVE MODE: Dynamic Epsilon ENABLEED (Sensitivity to Drawdown active)")
        else:
            self.logger.info("[INFO] BACKTEST/OTHER: Dynamic Epsilon DISABLED (Fixed to base values)")

        # Q-normalization config (snake_case)
        self.q_normalization: Dict[str, Dict[str, float]] = self.ensemble_cfg.get('q_normalization', {})
        # Update interval for ensemble logic (avoid recalculating too often)
        self.update_interval_sec: int = self.ensemble_cfg.get('update_interval_sec', 60)
        self.config_update_interval: int = self.ensemble_cfg.get(
            'q_update_interval', config.get('q_update_interval', 14400)
        )

        self.logger.info(
            f"[REGIME] Global Filter (BTC {self.informative_timeframe_global}): {'ON' if self.use_global_regime_filter else 'OFF'} | "
            f"Local Filter (Pair {self.timeframe}): {'ON' if self.use_local_regime_filter else 'OFF'}"
        )

        self.logger.info(
            f"[ENSEMBLE] Ensemble Config: EpsL={self.epsilon_threshold_long} | EpsS={self.epsilon_threshold_short} | "
            f"UpdateInterval={self.config_update_interval}s"
        )

        if not self.can_short:
            logger.warning("⚠️ WARNING: can_short is False! Short signals will be ignored.")

        # --- СИНХРОНИЗАЦИЯ ПАРАМЕТРОВ С КОНФИГОМ ---
        # ВАЖНО: Мы больше не перетираем значения из конфига принудительно, 
        # чтобы параметры Hyperopt в теле класса имели приоритет.
        
        # Память для логирования сигналов (чтобы не спамить каждую минуту)
        self._last_logged_signal = {}

        logger.info("=" * 60)
        logger.info("✅ 2+2 ENSEMBLE READY FOR TRADING")
        logger.info("✅ 2+2 ENSEMBLE READY FOR TRADING (Agents will be loaded on first use)")
        logger.info("=" * 60)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Исключаем объекты, которые нельзя пиклить (блокировки, потоки, логгеры)
        state.pop('cache_lock', None)
        state.pop('_global_regime_lock', None)
        state.pop('logger', None)
        state.pop('_last_logged_signal', None)

        # --- NEW: Explicitly remove agent instances before pickling ---
        # These will be lazy-loaded in the worker process.
        state.pop('_long_1_agent', None)
        state.pop('_long_2_agent', None)
        state.pop('_short_1_agent', None)
        state.pop('_short_2_agent', None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Восстанавливаем объекты в каждом воркере Hyperopt
        self.logger = logging.getLogger(__name__)
        self.cache_lock = threading.Lock()
        self._global_regime_lock = threading.Lock()
        self._last_logged_signal = {}
        
        # --- NEW: Ensure agent attributes are reset to None in the new process ---
        # This guarantees that the lazy-loading properties will trigger correctly.
        self._long_1_agent = None
        self._long_2_agent = None
        self._short_1_agent = None
        self._short_2_agent = None

    def _find_config_file(self, dir_path: Path):
        for file in dir_path.glob("*.py"):
            if "alpha" in file.name or "config" in file.name:
                return file
        return None
    
    def _load_py_config(self, file_path: Path):
        import importlib.util
        try:
            from pydantic import ValidationError  # type: ignore
        except ImportError:
            class ValidationError(Exception): pass  # type: ignore

        if not hasattr(importlib, 'util'):
            return None

        spec = importlib.util.spec_from_file_location("mod_cfg", file_path)  # type: ignore
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {file_path}")
            
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        
        # Хак для обратной совместимости: 
        # Если в загружаемом файле есть обращение к несуществующим полям Pydantic,
        # нам нужно это перехватить. 
        try:
            spec.loader.exec_module(mod)  # type: ignore
        except ValueError as e:
            logger.error(f"❌ Config loading failed: {e}. Attempting to bypass Pydantic validation...")
            # Если критично — здесь можно динамически добавить поле в PathConfig через setattr
            raise e

        return mod.cfg
    
    def _create_agent_from_config(self, cfg, mirror_mode=False):
        agent = D3QN_PER_Agent(
            state_shape=cfg.seq.state_shape,
            action_dim=cfg.market.num_actions,
            cnn_maps=cfg.model.cnn_maps,
            cnn_kernels=cfg.model.cnn_kernels,
            cnn_strides=cfg.model.cnn_strides,
            cnn_dilations=cfg.model.cnn_dilations,
            dense_val=cfg.model.dense_val,
            dense_adv=cfg.model.dense_adv,
            additional_feats=cfg.model.additional_feats,
            dropout_model=cfg.model.dropout_p,
            device=self.device,
            gamma=cfg.rl.gamma,
            learning_rate=cfg.rl.lr,
            batch_size=cfg.rl.batch_size,
            buffer_size=cfg.per.buffer_size,
            target_update_freq=cfg.rl.target_update_freq,
            train_start=cfg.rl.train_start,
            per_alpha=cfg.per.per_alpha,
            per_beta_start=cfg.per.per_beta_start,
            per_beta_frames=cfg.per.per_beta_frames,
            eps_start=cfg.eps.eps_start,
            eps_end=cfg.eps.eps_end,
            eps_frames=cfg.eps.eps_decay_frames,
            epsilon=0.0,
            max_gradient_norm=cfg.rl.max_gradient_norm
        )
        agent.mirror_mode = mirror_mode
        return agent
    
    # --- NEW: LAZY LOADING PROPERTIES FOR AGENTS ---

    @property
    def long_1_agent(self) -> Optional[D3QN_PER_Agent]:
        if not self.enable_long_1: return None
        if self._long_1_agent is None:
            self.logger.info("Lazily loading agent: LONG_1")
            self._long_1_agent = self._create_agent_from_config(self.cfg_long_1, mirror_mode=False)
            self._load_weights(self._long_1_agent, self.long_1_model_pth, "LONG_1")
            self._long_1_agent.policy_net.eval()
            for param in self._long_1_agent.policy_net.parameters():
                param.requires_grad = False
        return self._long_1_agent

    @property
    def long_2_agent(self) -> Optional[D3QN_PER_Agent]:
        if not self.enable_long_2: return None
        if self._long_2_agent is None:
            self.logger.info("Lazily loading agent: LONG_2")
            self._long_2_agent = self._create_agent_from_config(self.cfg_long_2, mirror_mode=False)
            self._load_weights(self._long_2_agent, self.long_2_model_pth, "LONG_2")
            self._long_2_agent.policy_net.eval()
            for param in self._long_2_agent.policy_net.parameters():
                param.requires_grad = False
        return self._long_2_agent

    @property
    def short_1_agent(self) -> Optional[D3QN_PER_Agent]:
        if not self.enable_short_1: return None
        if self._short_1_agent is None:
            self.logger.info("Lazily loading agent: SHORT_1")
            self._short_1_agent = self._create_agent_from_config(self.cfg_short_1, mirror_mode=self.short_1_is_mirror)
            self._load_weights(self._short_1_agent, self.short_1_model_pth, "SHORT_1")
            self._short_1_agent.policy_net.eval()
            for param in self._short_1_agent.policy_net.parameters():
                param.requires_grad = False
        return self._short_1_agent

    @property
    def short_2_agent(self) -> Optional[D3QN_PER_Agent]:
        if not self.enable_short_2: return None
        if self._short_2_agent is None:
            self.logger.info("Lazily loading agent: SHORT_2")
            self._short_2_agent = self._create_agent_from_config(self.cfg_short_2, mirror_mode=self.short_2_is_mirror)
            self._load_weights(self._short_2_agent, self.short_2_model_pth, "SHORT_2")
            self._short_2_agent.policy_net.eval()
            for param in self._short_2_agent.policy_net.parameters():
                param.requires_grad = False
        return self._short_2_agent

    # -------------------------------------------------

    def _load_weights(self, agent, path, name):
        onnx_path = Path(str(path).replace('.pth', '.onnx'))
        try:
            if onnx_path.exists():
                try:
                    # Оптимизация под AMD Ryzen 9 5900HX (8 cores / 16 threads)
                    # Так как инференс строго последовательный, отдаем графу максимум логических ядер
                    sess_options = ort.SessionOptions()
                    # Используем (Threads - 2) оставляя ресурсы для ОС и Freqtrade event loop
                    cpu_threads = self.config.get('cpu_threads', 12) 
                    sess_options.intra_op_num_threads = max(1, cpu_threads)
                    sess_options.inter_op_num_threads = 1 
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                    agent.ort_session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
                    self.logger.info(f"🚀 {name} ONNX session loaded from {onnx_path}")

                    # Пропускаем загрузку весов PyTorch в память, если успешно загружен ONNX
                    return
                except Exception as e:
                    self.logger.error(f"❌ Failed to load ONNX model for {name}: {e}")
                    agent.ort_session = None
            else:
                self.logger.warning(f"⚠️ ONNX model not found for {name} at {onnx_path}. Falling back to PyTorch.")
                agent.ort_session = None

            # Загружаем PyTorch только как fallback
            agent.load_model(str(path))
            agent.policy_net.eval()
            self.logger.info(f"✅ {name} PyTorch fallback loaded from {path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load {name} Agent: {e}")
            raise e

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Z-score нормализация (как в обучении)
        zscore_window = 90
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # 2. Quote Volume для фильтрации неликвида (РАССЧИТЫВАТЬ ДО ЛОГАРИФМИРОВАНИЯ)
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['close']
        dataframe['quote_volume_sma'] = dataframe['quote_volume'].rolling(window=1440, min_periods=200).mean()
        dataframe['quote_volume_sma'] = dataframe['quote_volume_sma'].fillna(0)

        # --- DYNAMIC VOLUME FLOW METRICS ---
        vol_win = int(self.vol_window.value)
        cvd_win = int(self.cvd_window.value)

        high_low_range = dataframe['high'] - dataframe['low']
        buy_pressure = np.where(high_low_range > 0, (dataframe['close'] - dataframe['low']) / high_low_range, 0.5)

        dataframe['buy_vol'] = dataframe['volume'] * buy_pressure
        dataframe['sell_vol'] = dataframe['volume'] * (1.0 - buy_pressure)

        # Применяем динамическое окно для SMA
        dataframe['vol_sma_dyn'] = dataframe['volume'].rolling(window=vol_win, min_periods=max(1, vol_win//2)).mean()
        dataframe['surge_ratio'] = dataframe['volume'] / (dataframe['vol_sma_dyn'] + 1e-8)

        dataframe['buy_vol_pct'] = (dataframe['buy_vol'] / (dataframe['volume'] + 1e-8)) * 100
        dataframe['sell_vol_pct'] = (dataframe['sell_vol'] / (dataframe['volume'] + 1e-8)) * 100

        # CVD с динамическим окном
        dataframe['cvd'] = (dataframe['buy_vol'] - dataframe['sell_vol']).cumsum()
        dataframe['cvd_ma'] = dataframe['cvd'].rolling(window=cvd_win, min_periods=max(1, cvd_win//2)).mean()

        dataframe['gap_pct_long'] = dataframe['buy_vol'] / (dataframe['sell_vol'] + 1e-8)
        dataframe['gap_pct_short'] = dataframe['sell_vol'] / (dataframe['buy_vol'] + 1e-8)

        # ATR для динамического стоп-лосса (Native pandas, safe for CPU inference)
        atr_win = int(self.atr_period.value)
        high_low = dataframe['high'] - dataframe['low']
        high_close = (dataframe['high'] - dataframe['close'].shift()).abs()
        low_close = (dataframe['low'] - dataframe['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        dataframe['atr'] = tr.rolling(atr_win).mean().fillna(0.0)

        # Log-transform volume to match training distribution
        dataframe['volume'] = np.log1p(dataframe['volume'])
        
        for col in ohlcv_cols:
            rolling = dataframe[col].rolling(window=zscore_window, min_periods=zscore_window)
            mean = rolling.mean()
            std = rolling.std(ddof=0)
            # Z-score: numerical stability epsilon updated to 1e-6
            dataframe[f'{col}_z'] = (dataframe[col] - mean) / (std + 1e-6)

        # Заполняем NaN нулями (начало датафрейма), чтобы модель не получала inf/nan
        z_cols = [f'{col}_z' for col in ohlcv_cols]
        dataframe[z_cols] = dataframe[z_cols].fillna(0.0)

        # 2. EMA Режимы (Вместо Supertrend)
        if hasattr(self, 'dp') and getattr(self, 'dp', None) is not None:
            ema_period = int(self.ema_fast_period.value)

            # --- LOCAL REGIME (Current Pair 1m) ---
            # Рассчитываем быструю EMA для фильтрации входов
            dataframe['ema_fast'] = dataframe['close'].ewm(span=ema_period, adjust=False).mean()
            dataframe['st_regime_local'] = np.where(dataframe['close'] > dataframe['ema_fast'], 1, -1)

            # --- GLOBAL REGIME (Current Pair Higher TF) ---
            try:
                # Используем информативный таймфрейм из параметра
                inf_tf_global = self.informative_timeframe_global
                
                # Используем глобальный период EMA
                global_ema_p = int(self.global_ema_period.value)

                # Теперь используем ТЕКУЩУЮ ПАРУ вместо BTC
                pair_name = metadata['pair']
                higher_tf_df = self.dp.get_pair_dataframe(pair_name, inf_tf_global)
                
                if higher_tf_df is not None and not higher_tf_df.empty:
                    # Рассчитываем EMA для текущей пары на старшем таймфрейме
                    higher_tf_df['ema_higher'] = higher_tf_df['close'].ewm(span=global_ema_p, adjust=False).mean()
                    
                    # Создаем временный DF для мерджа
                    merge_df = higher_tf_df[['date']].copy()
                    merge_df['st_regime_global'] = np.where(higher_tf_df['close'] > higher_tf_df['ema_higher'], 1, -1)
                    
                    # Debug print (только для первой пары, чтобы не спамить)
                    if metadata['pair'] == self.dp.current_whitelist()[0]:
                        print(f"DEBUG: Calculated Global Regime for {metadata['pair']} on {inf_tf_global}. Rows: {len(merge_df)}")

                    # Динамическая подстройка TZ
                    main_tz = dataframe['date'].dt.tz
                    if merge_df['date'].dt.tz != main_tz:
                        merge_df = merge_df.copy()
                        if main_tz is None:
                            merge_df['date'] = merge_df['date'].dt.tz_localize(None)
                        else:
                            if merge_df['date'].dt.tz is None:
                                merge_df['date'] = merge_df['date'].dt.tz_localize(main_tz)
                            else:
                                merge_df['date'] = merge_df['date'].dt.tz_convert(main_tz)

                    dataframe = merge_informative_pair(
                        dataframe, merge_df, 
                        self.timeframe, inf_tf_global, ffill=True
                    )
                    
                    # Извлекаем из колонки с суффиксом (напр. st_regime_global_15m)
                    inf_col = f"st_regime_global_{inf_tf_global}"
                    if inf_col in dataframe.columns:
                        dataframe['st_regime_global'] = dataframe[inf_col].fillna(0).astype(np.int8)
            except Exception as e:
                self.logger.warning(f"Global regime ({metadata['pair']} EMA {inf_tf_global}) calculation failed: {e}")

        return dataframe

    def informative_pairs(self):
        """
        Информативные пары:
        1. Все пары из whitelist на старшем таймфрейме (для "глобального" фильтра)
        """
        pairs: List[Any] = []
        if hasattr(self, 'dp') and getattr(self, 'dp', None) is not None:
            try:
                pairs = self.dp.current_whitelist()  # type: ignore
            except Exception:
                pairs = []
        
        if not pairs:
            pairs = self.config.get('exchange', {}).get('pair_whitelist', [])

        # Формируем список: (pair, global_ema_timeframe) для всех пар в whitelist
        # Это нужно, чтобы в populate_indicators был доступен старший ТФ для каждой пары
        inf_tf_global = self.informative_timeframe_global
        info_list = [(pair, inf_tf_global) for pair in pairs]
        
        # BTC/USDT всегда полезен, добавляем его если его нет (хотя он обычно в whitelist)
        btc_pair = 'BTC/USDT:USDT'
        if not any(p == btc_pair for p, tf in info_list):
            info_list.append((btc_pair, inf_tf_global))
            
        return info_list
    
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Вход лимитками для сбора спреда и снижения комиссии (Maker fee)."""
        if side == 'long':
            return proposed_rate * (1.0 - self.entry_discount_pct.value)
        else:
            return proposed_rate * (1.0 + self.entry_discount_pct.value)

    def check_entry_timeout(self, pair: str, trade: Trade, order: dict,
                            current_time: datetime, **kwargs) -> bool:
        """Отмена ордера, если он не исполнился за 3 свечи (180 секунд), чтобы не морозить слот."""
        if trade.open_date_utc and (current_time - trade.open_date_utc).total_seconds() > 180:
            self.logger.info(f"[TIMEOUT] Canceling unfilled limit entry for {pair}")
            return True
        return False
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """
        Умный контроль выходов:
        1. Soft Time-Stop: Освобождение слота от мертвых сделок.
        2. "Emergency Alpha Stop": Если модели уверены в развороте (согласно rl_exit_threshold)
           И убыток уже ощутимый (> emergency_exit_threshold), выходим не дожидаясь стопа.
        """
        # --- 1. Soft Time-Stop ---
        trade_open_date = getattr(trade, 'open_date_utc', None)
        if trade_open_date is not None:
            duration_min = (current_time - trade_open_date).total_seconds() / 60.0
            # Если сидим дольше 60 минут и профит ниже 0.5% (около нуля или убыток)
            if duration_min >= 60 and current_profit < 0.005:
                return "time_opportunity_cost"

        # --- 2. Emergency Alpha Stop ---
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]

        # Используем параметры стратегии для экстренного выхода
        emergency_loss = self.emergency_exit_threshold.value

        # Порог голосов для выхода (Alpha Decay)
        exit_long_thresh = self.rl_exit_long_threshold.value
        exit_short_thresh = self.rl_exit_short_threshold.value

        if trade.is_short:
            # Для шорта: выход при появлении голосов в Лонг
            votes_long = last_candle.get('votes_long', 0)
            if votes_long >= exit_long_thresh:
                if current_profit < emergency_loss:
                    self.logger.info(f"🚨 [ALPHA STOP] {pair} SHORT: Profit {current_profit:.2%} < {emergency_loss:.2%} with {votes_long} Long votes. EXITING.")
                    return "emergency_alpha_exit_short"
                else:
                    # Логируем, что сигнал есть, но убыток еще не достиг порога
                    if self.config.get('runmode') in ('live', 'dry_run'):
                        self.logger.info(f"⏳ [ALPHA STOP] {pair} SHORT: Reversal signal exists ({votes_long} Long votes) but profit {current_profit:.2%} > threshold {emergency_loss:.2%}. Waiting.")
        else:
            # Для лонга: выход при появлении голосов в Шорт
            votes_short = last_candle.get('votes_short', 0)
            if votes_short >= exit_short_thresh:
                if current_profit < emergency_loss:
                    self.logger.info(f"🚨 [ALPHA STOP] {pair} LONG: Profit {current_profit:.2%} < {emergency_loss:.2%} with {votes_short} Short votes. EXITING.")
                    return "emergency_alpha_exit_long"
                else:
                    # Логируем, что сигнал есть, но убыток еще не достиг порога
                    if self.config.get('runmode') in ('live', 'dry_run'):
                        self.logger.info(f"⏳ [ALPHA STOP] {pair} LONG: Reversal signal exists ({votes_short} Short votes) but profit {current_profit:.2%} > threshold {emergency_loss:.2%}. Waiting.")

        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        d0_val = self.d0.value
        d_min_val = self.d_min.value
        hysteresis_val = self.hysteresis.value
        FEE_BUF = 0.0008
        
        p = current_profit
        trade_id = trade.id
        
        if trade_id not in self.tsl_memory:
            self.tsl_memory[trade_id] = -999.0
        
        last_p = self.tsl_memory[trade_id]
        
        if p >= (last_p + hysteresis_val):
            self.tsl_memory[trade_id] = p
        
        # Используем запомненное значение (ступенчатое), чтобы гистерезис работал
        calc_p = self.tsl_memory[trade_id]

        # НЕЛИНЕЙНЫЙ ТРЕЙЛИНГ (агрессивность задается через tsl_exponent)
        if calc_p <= FEE_BUF:
            d_eff = d0_val
        else:
            p_factor = calc_p - FEE_BUF
            # Целевой профит, при котором отступ сужается до d_min
            p_target_val = self.p_target.value
            # Нормализуем (0..1)
            p_norm = min(1.0, p_factor / p_target_val) 
            # Используем оптимизируемую степень
            exponent = float(self.tsl_exponent.value)
            d_eff = d0_val - (d0_val - d_min_val) * (p_norm**exponent)
            d_eff = max(d_min_val, d_eff)
        
        # --- ATR DYNAMIC FLOOR (CORRECTED) ---
        try:
            # Извлекаем ATR из актуального фрейма
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty and 'atr' in dataframe.columns:
                atr_val = dataframe['atr'].iloc[-1]
                
                # Коэффициенты из параметров
                multiplier = float(self.atr_multiplier.value)
                hard_stop = float(self.d0.value)
                
                # Рассчитываем динамическую дистанцию (положительное число)
                # Это "пол" (floor), ниже которого стоп не должен опускаться при высокой волатильности
                atr_dist = (atr_val * multiplier) / current_rate
                
                # Логика: Выбираем МАКСИМАЛЬНУЮ дистанцию (самый широкий стоп) между TSL и ATR,
                # чтобы дать алгоритму дышать, НО ограничиваем её хард-стопом d0.
                
                # 1. Выбираем более широкий стоп (max от положительных дистанций)
                target_dist = max(d_eff, atr_dist)
                
                # 2. Ограничиваем хард-стопом (не шире d0)
                final_dist = min(target_dist, hard_stop)
                
                return -final_dist
        except Exception as e:
            self.logger.error(f"Error in ATR Dynamic Floor: {e}")
            pass

        return -d_eff

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str) -> Optional[torch.Tensor]:
        # Default window is 90 (5 channels * 90 = 450 history features)
        window = 90 
        
        # Safe access to model configs
        cfg = None
        if side == "LONG":
            cfg = self.cfg_long_1 if model_num == 1 else self.cfg_long_2
        elif side == "SHORT":
            cfg = self.cfg_short_1 if model_num == 1 else self.cfg_short_2
            
        if cfg and hasattr(cfg, 'seq') and cfg.seq:
            window = getattr(cfg.seq, "agent_history_len", 90)

        if len(dataframe) < window:
            return None

        # Inversion logic (Mirror Mode)
        should_invert = False
        if side == "SHORT":
            if (model_num == 1 and self.short_1_is_mirror) or (model_num == 2 and self.short_2_is_mirror):
                should_invert = True

        if self.config.get('runmode') in ('live', 'dry_run'):
            if 'open_z' in dataframe.columns:
                last_window = dataframe[['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']].iloc[-window:].values.copy().astype(np.float32)
            else:
                raw_cols = ['open', 'high', 'low', 'close', 'volume']
                arr = dataframe[raw_cols].iloc[-180:].values.astype(np.float32)
                arr[:, 4] = np.log1p(arr[:, 4])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0, ddof=0) + 1e-6
                last_window = ((arr[-window:] - mean) / std).astype(np.float32)

            if should_invert:
                # 1. Инвертируем ТОЛЬКО цены (Open, High, Low, Close -> индексы 0, 1, 2, 3)
                last_window[:, :4] *= -1.0
                # 2. Свопаем колонки High и Low (индексы 1 и 2), т.к. после инверсии они поменялись ролями
                last_window[:, [1, 2]] = last_window[:, [2, 1]]
                # Объем (индекс 4) остается без изменений
            
            img_flat = last_window.T.flatten()
            add_feats = np.zeros(4, dtype=np.float32)
            full_input = np.concatenate([img_flat, add_feats])
            return np.expand_dims(full_input, axis=0)
        else:
            # BATCH INFERENCE (Full Backtest)
            cols = ['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']
            z_data = dataframe[cols].values.astype(np.float32)
            try:
                windows = sliding_window_view(z_data, window_shape=(window, 5)).squeeze(1).copy()
            except Exception as e:
                self.logger.error(f"Sliding window failed: {e}")
                return None

            if should_invert:
                # 1. Инвертируем цены в батче
                windows[:, :, :4] *= -1.0
                # 2. Свопаем High и Low в батче
                windows[:, :, [1, 2]] = windows[:, :, [2, 1]]
            
            img_batch = windows.transpose(0, 2, 1).reshape(len(windows), -1)
            add_feats = np.zeros((len(windows), 4), dtype=np.float32)
            full_input = np.concatenate([img_batch, add_feats], axis=1)
            return full_input

    def get_model_input_cached(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str):
        """
        Кэширующая версия get_model_input
        Ключ кэша = (pair, last_candle_timestamp, side, model_num)
        """
        # Генерируем ключ кэша
        last_ts = dataframe.iloc[-1]['date'] if 'date' in dataframe.columns else dataframe.index[-1]
        # Оптимизация: используем int (value) вместо медленного str()
        ts_val = last_ts.value if hasattr(last_ts, 'value') else hash(last_ts)
        cache_key = (pair, ts_val, side, model_num)
        
        with self.cache_lock:
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
        
        tensor = self.get_model_input(dataframe, pair, side, model_num, asset_name)
        
        with self.cache_lock:
            if len(self.feature_cache) >= self.cache_max_size:
                self.feature_cache.popitem(last=False)
            self.feature_cache[cache_key] = tensor
            self.feature_cache.move_to_end(cache_key)
        
        return tensor

    def _run_inference(self, tensors_and_agents):
        """
        Runs inference sequentially for a list of (arr, agent, name) tuples.
        Prioritizes ONNX Runtime if available, otherwise falls back to PyTorch.
        """
        results = {}

        valid_tasks = [t for t in tensors_and_agents if t[0] is not None]

        for arr, agt, nm in valid_tasks:
            try:
                if hasattr(agt, 'ort_session') and agt.ort_session is not None:
                    ort_inputs = {agt.ort_session.get_inputs()[0].name: arr}
                    q_val = agt.ort_session.run(None, ort_inputs)[0]
                else:
                    with torch.no_grad():
                        t = torch.as_tensor(arr, device=self.device, dtype=torch.float32)
                        q_val = agt.policy_net(t).cpu().numpy()
                results[nm] = q_val
            except Exception as e:
                self.logger.error(f"Inference failed for {nm}: {e}")

        return results

    def _collect_adv_stats(self, name: str, adv_array: np.ndarray):
        """
        Сбор статистики advantage для автоподбора q_min/q_max.
        Вместо одного последнего значения используем все положительные
        значения из adv_array и храним короткое скользящее окно.
        """
        if name not in self.adv_history:
            # Примерно 5 минут истории:
            # при большом количестве пар метод вызывается очень часто,
            # поэтому 1280 элементов дают короткое, но репрезентативное окно.
            self.adv_history[name] = deque(maxlen=242)

        if adv_array is None or len(adv_array) == 0:
            return

        # Берем только положительные advantage (сигналы выше нуля)
        # и добавляем их в общую историю по данной модели.
        pos_arr = np.asarray(adv_array, dtype=float)
        pos_arr = pos_arr[pos_arr > 0.0]
        if pos_arr.size > 0:
            self.adv_history[name].extend(pos_arr.tolist())

    def update_normalization_config(self):
        try:
            config_path = self.project_root / "user_data/config_rl4z.json"
            if not config_path.exists():
                self.logger.warning(f"⚠️ Config update skipped: {config_path} not found")
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            updated = False
            norm_cfg = data.get('rl_ensemble', {}).get('q_normalization', {})
            
            for name, history in self.adv_history.items():
                if len(history) < 100: 
                    self.logger.info(f"⏳ {name}: Insufficient history for Q-update ({len(history)}/100)")
                    continue # Мало данных
                
                arr = np.array(history)
                # Берем только положительные значения (сигналы)
                pos_arr = arr[arr > 0]
                if len(pos_arr) < 50: 
                    continue
                
                # Считаем перцентили: q_min (шум) и q_max (пик)
                new_min = float(round(np.percentile(pos_arr, 15), 6))
                new_max = float(round(np.percentile(pos_arr, 99), 6))
                
                if name not in norm_cfg: norm_cfg[name] = {}
                norm_cfg[name]['q_min'] = new_min
                norm_cfg[name]['q_max'] = new_max
                updated = True
                self.logger.info(f"⚖️ Auto-tuned {name}: q_min={new_min}, q_max={new_max}")
            
            if updated:
                if 'rl_ensemble' not in data: data['rl_ensemble'] = {}
                data['rl_ensemble']['q_normalization'] = norm_cfg
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                self.q_normalization = norm_cfg
                self.logger.info(f"💾 Config saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to auto-tune config: {e}")

    def calculate_current_price(self, trade: Trade, current_time: datetime) -> float:
        """Безопасное получение текущей цены с защитой от lookahead bias"""
        if self.config.get("runmode") == "backtest":
            if hasattr(self, 'dp') and self.dp is not None:
                try:
                    dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                    mask = dataframe['date'] <= current_time
                    if mask.any():
                        return dataframe.loc[mask, 'close'].iloc[-1]
                except Exception:
                    pass
            return trade.open_rate
        else:
            return self.dp.current_whitelist_price(trade.pair)

    def _get_pnl_from_freqtrade(self, current_time: Optional[datetime] = None) -> tuple:
        """
        Получает ТОЛЬКО Unrealized PnL (открытые позиции)
        для мгновенной реакции на изменение рынка
        Возвращает (pnl_long_usdt, pnl_short_usdt)
        """
        try:
            # === ОТКРЫТЫЕ ПОЗИЦИИ (Unrealized PnL) ===
            open_trades = Trade.get_open_trades()

            pnl_long_open: float = 0.0
            pnl_short_open: float = 0.0

            for t in open_trades:
                try:
                    if t.close_rate_requested:
                        profit_usdt = t.calc_profit(rate=t.close_rate_requested)
                    else:
                        if current_time and self.config.get("runmode") == "backtest":
                            current_rate = self.calculate_current_price(t, current_time)
                        elif hasattr(self, 'dp') and getattr(self, 'dp', None) is not None:
                            try:
                                dataframe, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                                if not dataframe.empty:
                                    current_rate = dataframe['close'].iloc[-1]
                                else:
                                    current_rate = t.open_rate
                            except Exception:
                                current_rate = t.open_rate
                        else:
                            current_rate = t.open_rate
                            
                        # Consistent symmetric PnL calculation across both sides
                        direction = -1.0 if t.is_short else 1.0
                        position = t.amount * t.open_rate
                        profit_usdt = (((current_rate - t.open_rate) / t.open_rate) * direction) * position
                        
                except Exception as e:
                    logger.debug(f"Failed to calc profit for {t.pair}: {e}")
                    profit_usdt = 0.0

                if t.is_short:
                    pnl_short_open += profit_usdt
                else:
                    pnl_long_open += profit_usdt

            # Возвращаем ТОЛЬКО Unrealized PnL
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(
                    f"PnL (Unrealized Only): "
                    f"Long Open={pnl_long_open:.2f} | "
                    f"Short Open={pnl_short_open:.2f}"
                )

            return pnl_long_open, pnl_short_open

        except Exception as e:
            self.logger.error(f"Failed to calculate PnL: {e}")
            return 0.0, 0.0

    def _update_dynamic_epsilon(self, current_time: Optional[datetime] = None) -> None:
        """
        Update effective epsilon based on current equity drawdown.
        In calibration mode, dynamic epsilon is disabled.
        """
        if getattr(self, "calibration_mode", False) or not self.ensemble_cfg.get('enable_dynamic_epsilon', False):
            self.epsilon_threshold_eff_long = float(self.epsilon_threshold_long)
            self.epsilon_threshold_eff_short = float(self.epsilon_threshold_short)
            self.epsilon_threshold_eff = 0.5 * (
                self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short
            )
            return

        if self.config.get("runmode") not in ("live", "dry_run", "backtest"):
            return

        try:
            pnl_long, pnl_short = self._get_pnl_from_freqtrade(current_time)
            equity_long = pnl_long
            equity_short = pnl_short

            # Reset dynamic epsilon if there are no open trades
            try:
                open_trades = Trade.get_open_trades()
            except Exception:
                open_trades = []

            if not open_trades:
                # Вне рынка: сбрасываем состояние drawdown и возвращаемся к базовому epsilon
                self.equity_max_long = 0.0
                self.equity_max_short = 0.0
                self.epsilon_threshold_eff_long = self.epsilon_threshold_long
                self.epsilon_threshold_eff_short = self.epsilon_threshold_short
                self.epsilon_threshold_eff = 0.5 * (
                    self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short
                )
                return

            # Initialize equity_max_long/short on first run
            if self.equity_max_long <= 0.0:
                self.equity_max_long = equity_long
            if self.equity_max_short <= 0.0:
                self.equity_max_short = equity_short

            # Track maximum observed equity per side
            self.equity_max_long = max(self.equity_max_long, equity_long)
            self.equity_max_short = max(self.equity_max_short, equity_short)

            # Relative drawdown per side in [0.0, 1.0]
            dd_long = (self.equity_max_long - equity_long) / self.equity_max_long if self.equity_max_long > 0.0 else 0.0
            dd_short = (self.equity_max_short - equity_short) / self.equity_max_short if self.equity_max_short > 0.0 else 0.0

            dd_long = max(0.0, min(dd_long, 1.0))
            dd_short = max(0.0, min(dd_short, 1.0))

            # Используем настраиваемый коэффициент агрессии
            k = self.ensemble_cfg.get('dynamic_epsilon_k', self.dd_aggression_k.value)
            min_eps = self.ensemble_cfg.get('min_epsilon', 0.1)

            # Scale dynamic targets around side-specific base thresholds
            epsilon_target_long = self.epsilon_threshold_long * (1.0 + k * dd_long)
            epsilon_target_short = self.epsilon_threshold_short * (1.0 + k * dd_short)

            alpha = 0.4
            # Smooth update via EMA
            self.epsilon_threshold_eff_long = (1.0 - alpha) * self.epsilon_threshold_eff_long + alpha * epsilon_target_long
            self.epsilon_threshold_eff_short = (1.0 - alpha) * self.epsilon_threshold_eff_short + alpha * epsilon_target_short

            # Clamp effective thresholds
            self.epsilon_threshold_eff_long = float(min(max(self.epsilon_threshold_eff_long, min_eps), 1.0))
            self.epsilon_threshold_eff_short = float(min(max(self.epsilon_threshold_eff_short, min_eps), 1.0))

            self.epsilon_threshold_eff = 0.5 * (
                self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short
            )
        except Exception as e:
            self.logger.warning(f"Dynamic epsilon update failed: {e}. Falling back to base epsilon.")
            self.epsilon_threshold_eff = self.epsilon_threshold
            self.epsilon_threshold_eff_long = self.epsilon_threshold_long
            self.epsilon_threshold_eff_short = self.epsilon_threshold_short

    def _update_slot_allocation(self, current_time: datetime) -> None:
        """
        Динамическое перераспределение слотов на основе PnL из FreqTrade
        """
        if not self.dynamic_slots_enabled or self.total_slots <= 0:
            return

        pnl_long, pnl_short = self._get_pnl_from_freqtrade()

        # === УЛУЧШЕННАЯ ЛОГИКА V2 ===
        if pnl_long > 0 and pnl_short < 0:
            # Пропорциональное наказание убыточного направления
            profit_long = pnl_long
            loss_short = abs(pnl_short)
            total = profit_long + loss_short

            # Aggression factor as root (1/3) makes penalty curve convex -> harsher punishment
            penalty_ratio = (loss_short / total) ** (1.0 / self.aggression_factor)
            long_ratio = 0.85 + 0.10 * penalty_ratio
            reason = f"Long profitable (+{pnl_long:.0f}), Short losing (-{loss_short:.0f})"

        elif pnl_short > 0 and pnl_long < 0:
            # Зеркально
            profit_short = pnl_short
            loss_long = abs(pnl_long)
            total = profit_short + loss_long

            penalty_ratio = (loss_long / total) ** (1.0 / self.aggression_factor)
            long_ratio = 0.15 - 0.10 * penalty_ratio
            reason = f"Short profitable (+{pnl_short:.0f}), Long losing (-{loss_long:.0f})"

        elif pnl_long > 0 and pnl_short > 0:
            long_ratio = pnl_long / (pnl_long + pnl_short)
            reason = f"Both profitable (L:{pnl_long:.0f} S:{pnl_short:.0f})"
        elif pnl_long < 0 and pnl_short < 0:
            # ОБА УБЫТОЧНЫ → ИНВЕРТИРОВАННАЯ ПРОПОРЦИЯ
            loss_long = abs(pnl_long)
            loss_short = abs(pnl_short)
            total_loss = loss_long + loss_short

            if total_loss > 0:
                # Инвертируем: большему убытку - меньше слотов
                base_ratio = loss_short / total_loss
                long_ratio = base_ratio ** (1.0 / self.aggression_factor)
                reason = f"Both losing - inverse allocation (L:-{loss_long:.0f} S:-{loss_short:.0f})"
            else:
                long_ratio = 0.5
                reason = "Both at zero"
        else:
            long_ratio = 0.5
            reason = "One side at zero"

        # Ограничиваем в разумных пределах
        long_ratio = max(0.05, min(0.95, long_ratio))

        available = self.total_slots - 2 * self.min_slots_per_side
        if available < 0:
            # Защита если min_slots_per_side слишком велик
            self.max_long_slots = self.total_slots // 2
            self.max_short_slots = self.total_slots - self.max_long_slots
        else:
            self.max_long_slots = self.min_slots_per_side + int(available * long_ratio)
            self.max_short_slots = self.total_slots - self.max_long_slots

        self.slot_history.append((current_time, self.max_long_slots, self.max_short_slots, pnl_long, pnl_short))
        self.logger.info(f"SLOTS: L={self.max_long_slots} ({pnl_long:+.1f} USDT) | S={self.max_short_slots} ({pnl_short:+.1f} USDT) | {reason}")

    def _normalize_q_value(self, q_value: float, model_name: str) -> float:
        """
        Нормализует Q-value модели в диапазон [0, 1]
        Использует конфиг rl_ensemble.q_normalization[model_name]
        """
        norm_cfg = self.q_normalization
        cfg = norm_cfg.get(model_name, {})
        q_min = cfg.get('q_min', 0.0)
        q_max = cfg.get('q_max', q_min)

        if q_max <= q_min:
            self.logger.warning(f"⚠️ Degenerate Q stats for {model_name}: q_min={q_min}, q_max={q_max}. Model is likely a zombie.")
            return 0.0

        q_norm = (q_value - q_min) / (q_max - q_min)
        return np.clip(q_norm, 0.0, 1.0)

    def _compute_ensemble_decision(
        self,
        q_values: Dict[str, np.ndarray],
        idx: int,
        has_long: bool,
        has_short: bool
    ) -> Dict[str, Any]:
        """
        Алгоритм ансамбля: Голосование с порогом ε
        """
        # Always use the value from the Hyperoptable Parameter
        # Freqtrade handles default/config/hyperopt file automatically
        thresh_long = self.rl_long_threshold_opt.value
        thresh_short = self.rl_short_threshold_opt.value

        votes_long = 0
        votes_short = 0
        details = []

        # --- 1. Подсчет голосов LONG ---
        def _get_model_vote(name, action_idx):
            if name not in q_values: return 0, False, 0.0
            q_hold = q_values[name][idx, 0]
            q_action = q_values[name][idx, action_idx]
            adv = q_action - q_hold

            cfg = self.q_normalization.get(name, {})
            q_min = cfg.get('q_min', 0.0)
            q_max = cfg.get('q_max', q_min)  # fallback to q_min to avoid None comparisons

            if q_max is None or q_max <= q_min:
                self.logger.warning(f"[WARNING] Degenerate Q stats for {name}, excluding zombie model.")
                return 0, False, 0.0

            # [FIXED] Redundant "if adv <= q_min:" check removed. The full threshold calculation below now correctly uses epsilon.

            # Раздельный epsilon по направлению
            if name.startswith("long_"):
                eps_eff = self.epsilon_threshold_eff_long
            else:
                eps_eff = self.epsilon_threshold_eff_short

            thr = q_min + (q_max - q_min) * eps_eff if q_max > q_min else q_min
            norm = self._normalize_q_value(adv, name)

            if adv > thr:
                return 1, True, norm
            return 0, True, norm

        if self.enable_long_1:
            v, active, norm = _get_model_vote("long_1", 1)
            votes_long += v
            if v: details.append(f"L1({norm:.2f})")

        if self.enable_long_2:
            v, active, norm = _get_model_vote("long_2", 1)
            votes_long += v
            if v: details.append(f"L2({norm:.2f})")

        # --- 2. Подсчет голосов SHORT ---
        if self.enable_short_1:
            # For Mirror Models (and models trained with num_actions=3 but filter_direction=SHORT)
            # The entry action is always at index 1.
            action_idx = 1
            v, active, norm = _get_model_vote("short_1", action_idx)
            votes_short += v
            if v: details.append(f"S1({norm:.2f})")

        if self.enable_short_2:
            action_idx = 1 
            v, active, norm = _get_model_vote("short_2", action_idx)
            votes_short += v
            if v: details.append(f"S2({norm:.2f})")

        result: Dict[str, Any] = {
            'enter_long': 0,
            'enter_short': 0,
            'reason': f"Votes L:{votes_long}/{thresh_long} S:{votes_short}/{thresh_short} [{' '.join(details)}]",
        }

        if has_long or has_short:
            result['reason'] += " | Position exists"
            return result

        # --- 3. Логика принятия решения и вето ---
        long_signal = votes_long >= thresh_long
        short_signal = votes_short >= thresh_short

        # Вето срабатывает, если есть хотя бы один голос с противоположной стороны
        long_is_vetoed = self.enable_veto and votes_short > 0
        short_is_vetoed = self.enable_veto and votes_long > 0

        if long_signal and not short_signal and not long_is_vetoed:
            result['enter_long'] = 1
            result['reason'] += " | LONG Signal"
        elif short_signal and not long_signal and not short_is_vetoed and self.can_short:
            result['enter_short'] = 1
            result['reason'] += " | SHORT Signal"
        else:
            # Логируем причину, почему не вошли
            if long_signal and short_signal: result['reason'] += " | CONFLICT"
            elif long_signal and long_is_vetoed: result['reason'] += " | LONG Vetoed"
            elif short_signal and short_is_vetoed: result['reason'] += " | SHORT Vetoed"

        return result

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        # Проверки слотов, таймаутов и т.д. работают только в live/dry-run режимах.
        # В режиме бэктеста эта функция пропускается для ускорения и потому, что
        # методы `Trade.get_...` ведут себя иначе.
        if self.config.get('runmode') not in ('live', 'dry_run'):
            return True
        
        try:
            # --- 90-MINUTE DIRECTIONAL TIMEOUT ---
            # 1. Находим последнюю закрытую сделку по этой паре
            trades_query = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(False)])
            if hasattr(trades_query, 'order_by'):
                last_trade = trades_query.order_by(Trade.close_date.desc()).first()
            else:
                # Fallback for older freqtrade versions returning a list
                min_date = datetime.min.replace(tzinfo=timezone.utc)
                sorted_trades = sorted(trades_query, key=lambda x: x.close_date if x.close_date else min_date, reverse=True)
                last_trade = sorted_trades[0] if sorted_trades else None
            
            if last_trade and last_trade.close_date:
                c_date = last_trade.close_date
                if c_date.tzinfo is None:
                    c_date = c_date.replace(tzinfo=timezone.utc)

                minutes_since = (current_time - c_date).total_seconds() / 60.0
                # Smart Cooldown: Блокируем после ЛЮБОЙ сделки (win/loss)
                if minutes_since < 90:
                    last_side = "short" if last_trade.is_short else "long"
                    # Блокируем только если направление совпадает (Long после Long или Short после Short)
                    if last_side == side:
                        profit_str = f"P={last_trade.close_profit:.2%}" if last_trade.close_profit is not None else "P=N/A"
                        self.logger.info(f"⏳ TIMEOUT {pair}: Last {last_side} ({profit_str}) closed {minutes_since:.1f}m ago. Blocking new {side}.")
                        return False
            # -------------------------------------

            open_trades_q = Trade.get_trades([Trade.is_open.is_(True)])
            if hasattr(open_trades_q, 'all'):
                trades = open_trades_q.all()
            else:
                trades = open_trades_q

            # === ДИНАМИЧЕСКОЕ ОБНОВЛЕНИЕ СЛОТОВ ===
            if self.dynamic_slots_enabled:
                if self.slot_update_interval == 0:
                    # On-Demand mode: обновляем при каждом входе
                    self._update_slot_allocation(current_time)
                    self.last_slot_update = current_time
                else:
                    # Таймер mode: обновляем по интервалу
                    last_upd = self.last_slot_update
                    if last_upd is None or \
                       (current_time - last_upd).total_seconds() > self.slot_update_interval:
                        self._update_slot_allocation(current_time)
                        self.last_slot_update = current_time
            else:
                # Фиксированные лимиты (legacy)
                if self.total_slots > 0:
                    self.max_long_slots = self.total_slots // 2
                    self.max_short_slots = self.total_slots - self.max_long_slots
                else:
                    self.max_long_slots = 50
                    self.max_short_slots = 50

            current_shorts = sum(1 for t in trades if t.is_short)
            current_longs = sum(1 for t in trades if not t.is_short)

            if side == "long":
                if current_longs >= self.max_long_slots:
                    self.logger.info(f"[LIMIT] LONG LIMIT: {pair} {current_longs}/{self.max_long_slots}")
                    return False
            elif side == "short":
                if current_shorts >= self.max_short_slots:
                    self.logger.info(f"[LIMIT] SHORT LIMIT: {pair} {current_shorts}/{self.max_short_slots}")
                    return False
        except Exception as e:
            self.logger.error(f"Error in confirm_trade_entry: {e}")
            return True
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, sell_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        trade_id = trade.id
        self.tsl_memory.pop(trade_id, None)
        return True
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        OPTIMIZED ENSEMBLE ENTRY LOGIC с параллельным инференсом
        """
        if dataframe.empty:
            return dataframe

        # IMPULSE GUARD: Запрет на покупку вытянутых свечей (защита от хаев)
        dataframe['impulse_long_ok'] = (dataframe['close'] / dataframe['open']) < 1.005
        dataframe['impulse_short_ok'] = (dataframe['close'] / dataframe['open']) > 0.995

        # СИНХРОНИЗАЦИЯ: Обновляем базовые значения из параметров Hyperopt
        self.epsilon_threshold_long = float(self.rl_epsilon_long.value)
        self.epsilon_threshold_short = float(self.rl_epsilon_short.value)

        # Get the current time from the last candle
        current_time = dataframe.iloc[-1]['date']

        # --- 0. LOOKAHEAD VALIDATION (from patch) ---
        if self.config.get('runmode') == 'backtest':
            last_candle_time = dataframe['date'].max()
            if last_candle_time > current_time:
                self.logger.error(f"DETECTED LOOKAHEAD BIAS in {metadata['pair']}: Future data detected! "
                                 f"Last in DF: {last_candle_time} | Current: {current_time}")
                raise ValueError("Lookahead bias detected in backtest mode")

        # Update dynamic epsilon once per candle based on current equity drawdown
        self._update_dynamic_epsilon(current_time)

        # --- PERIODIC SLOT UPDATE & LOGGING ---
        # Ensure slots are logged regularly for the dashboard, even if no trades occur
        if self.dynamic_slots_enabled:
            try:
                # В бэктесте используем время свечи, в лайве можно системное, 
                # но время свечи универсальнее для логики стратегии.
                now_logic = current_time if self.config.get('runmode') == 'backtest' else datetime.now(timezone.utc)
                
                if self.last_slot_update is None:
                    # Force immediate update on first run
                    self.last_slot_update = now_logic - timedelta(days=1)
                
                if (now_logic - self.last_slot_update).total_seconds() > self.slot_update_interval:
                    self._update_slot_allocation(now_logic)
                    self.last_slot_update = now_logic
            except Exception as e:
                self.logger.error(f"Periodic slot update failed: {e}")

        # 1. Базовая защита
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        # 2. Regime-фильтр: Глобальный (BTC 15m) + Локальный (Asset 15m)
        allow_long = True
        allow_short = self.can_short

        # Detect backtest/deep_inference mode
        is_backtest = self.config.get('runmode') not in ['live', 'dry_run'] or self.config.get('deep_inference', False)

        # Live/Dry-run: Получаем значения из последней свечи и применяем фильтры
        if not is_backtest and (self.use_global_regime_filter or self.use_local_regime_filter):
            regime_global = 0
            regime_local = 0
            try:
                if self.use_global_regime_filter:
                    if 'st_regime_global' in dataframe.columns:
                        regime_global = dataframe['st_regime_global'].iloc[-1]
                        allow_long = allow_long and (regime_global > 0)
                        allow_short = allow_short and (regime_global < 0)
                    else:
                        self.logger.warning(f"Global regime filter enabled but 'st_regime_global' column missing. Disabling trades.")
                        allow_long = allow_short = False

                if self.use_local_regime_filter:
                    if 'st_regime_local' in dataframe.columns:
                        regime_local = dataframe['st_regime_local'].iloc[-1]
                        allow_long = allow_long and (regime_local > 0)
                        allow_short = allow_short and (regime_local < 0)
                    else:
                        self.logger.warning(f"Local regime filter enabled but 'st_regime_local' column missing. Disabling trades.")
                        allow_long = allow_short = False

                # Логирование режима
                g_str = "BULL" if regime_global > 0 else ("BEAR" if regime_global < 0 else "FLAT")
                l_str = "BULL" if regime_local > 0 else ("BEAR" if regime_local < 0 else "FLAT")
                g_used = "Y" if self.use_global_regime_filter else "N"
                l_used = "Y" if self.use_local_regime_filter else "N"
                
                if self.config.get('runmode') in ['live', 'dry_run']:
                    self.logger.info(
                        f"[REGIME] {metadata['pair']} Global:{g_str}({g_used}) | Local:{l_str}({l_used}) | "
                        f"ALLOW: {'LONG' if allow_long else ('SHORT' if allow_short else 'NONE')}"
                    )
            except Exception as e:
                self.logger.warning(f"Regime filter failed for {metadata.get('pair', '')}: {e}")
                allow_long = allow_short = False # Fail-Closed

        # 3. Оптимизация инференса
        if self.config.get('runmode') in ['live', 'dry_run']:
            # Оптимизация: берем ровно 90 свечей для 1 предсказания (вместо 91 для 2)
            df_input = dataframe.iloc[-90:].copy()
        else:
            # Backtest or Hyperopt mode: process the full dataframe
            df_input = dataframe
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
        
        # 3. ОПТИМИЗИРОВАННЫЙ инференс с кэшированием
        asset_name = metadata['pair'].split(':')[0].replace('/', '')
        
        # --- Q-VALUE CACHING FOR HYPEROPT ---
        last_date = dataframe.iloc[-1]['date']
        # Оптимизация: используем числовое значение даты вместо медленной строки для ключа кэша
        q_cache_key = (metadata['pair'], last_date.value if hasattr(last_date, 'value') else str(last_date))
        
        q_values = None
        with self.cache_lock:
            if q_cache_key in self.q_value_cache:
                q_values = self.q_value_cache[q_cache_key]
        
        if q_values is None:
            # (Inference logic follows...)
            # Если в кэше нет - считаем (Тяжелая операция)
            tensor_long_1 = self.get_model_input_cached(
                df_input, metadata['pair'], side="LONG", model_num=1, asset_name=asset_name
            ) if (self.enable_long_1 and allow_long) else None
            tensor_long_2 = self.get_model_input_cached(
                df_input, metadata['pair'], side="LONG", model_num=2, asset_name=asset_name
            ) if (self.enable_long_2 and allow_long) else None
            tensor_short_1 = self.get_model_input_cached(
                df_input, metadata['pair'], side="SHORT", model_num=1, asset_name=asset_name
            ) if (self.enable_short_1 and allow_short) else None
            tensor_short_2 = self.get_model_input_cached(
                df_input, metadata['pair'], side="SHORT", model_num=2, asset_name=asset_name
            ) if (self.enable_short_2 and allow_short) else None

            # Проверяем, что все ВКЛЮЧЕННЫЕ модели получили данные
            missing_data = False
            if self.enable_long_1 and allow_long and tensor_long_1 is None: missing_data = True
            if self.enable_long_2 and allow_long and tensor_long_2 is None: missing_data = True
            if self.enable_short_1 and allow_short and tensor_short_1 is None: missing_data = True
            if self.enable_short_2 and allow_short and tensor_short_2 is None: missing_data = True

            if missing_data:
                return dataframe

            # 4. ПАРАЛЛЕЛЬНЫЙ INFERENCE
            inference_tasks = []
            if self.enable_long_1 and allow_long and tensor_long_1 is not None:
                inference_tasks.append((tensor_long_1, self.long_1_agent, "long_1"))
            if self.enable_long_2 and allow_long and tensor_long_2 is not None:
                inference_tasks.append((tensor_long_2, self.long_2_agent, "long_2"))
            if self.enable_short_1 and allow_short and tensor_short_1 is not None:
                inference_tasks.append((tensor_short_1, self.short_1_agent, "short_1"))
            if self.enable_short_2 and allow_short and tensor_short_2 is not None:
                inference_tasks.append((tensor_short_2, self.short_2_agent, "short_2"))
                
            if not inference_tasks:
                return dataframe
            
            q_values = self._run_inference(inference_tasks)
            
            # Сохраняем в кэш
            with self.cache_lock:
                if len(self.q_value_cache) >= self.cache_max_size:
                    self.q_value_cache.popitem(last=False)
                self.q_value_cache[q_cache_key] = q_values
        
        # Определяем размер батча из первого доступного результата
        batch_size = next(iter(q_values.values())).shape[0]
        
        # 5. Получение действий с порогом уверенности (Q-Threshold)
        # Фильтруем слабые сигналы, где Q(Action) почти равно Q(Hold)

        def get_action_with_threshold(name: str, target_action: Optional[int] = None):
            if q_values is None or name not in q_values:
                return np.zeros(batch_size, dtype=int), np.zeros(batch_size)

            q = q_values[name]
            advantage = q[:, target_action] - q[:, 0] if target_action is not None else np.zeros(batch_size)
            
            # --- CORRECTED LOGIC ---
            cfg = self.q_normalization.get(name, {})
            q_min_val = cfg.get('q_min', 0.0)
            q_max_val = cfg.get('q_max', q_min_val)

            if name.startswith("long_"):
                eps_eff = self.epsilon_threshold_eff_long
            else:
                eps_eff = self.epsilon_threshold_eff_short
            
            # Calculate the threshold correctly using epsilon
            thr = q_min_val + (q_max_val - q_min_val) * eps_eff if q_max_val > q_min_val else q_min_val

            final_actions = np.where(advantage > thr, target_action, 0)
            # --- END CORRECTED LOGIC ---

            return final_actions, advantage

        action_long_1, adv_long_1 = get_action_with_threshold("long_1", target_action=1)
        action_long_2, adv_long_2 = get_action_with_threshold("long_2", target_action=1)
        
        # Mirror mode models (and all models in config trained as SHORT) 
        # use index 1 for their primary 'ENTRY' action
        action_short_1, adv_short_1 = get_action_with_threshold("short_1", target_action=1)
        action_short_2, adv_short_2 = get_action_with_threshold("short_2", target_action=1)

        # --- СБОР СТАТИСТИКИ ДЛЯ АВТОПОДБОРА ---
        if self.config.get('runmode') in ['live', 'dry_run']:
            if self.enable_long_1 and "long_1" in q_values: self._collect_adv_stats("long_1", adv_long_1)
            if self.enable_long_2 and "long_2" in q_values: self._collect_adv_stats("long_2", adv_long_2)
            if self.enable_short_1 and "short_1" in q_values: self._collect_adv_stats("short_1", adv_short_1)
            if self.enable_short_2 and "short_2" in q_values: self._collect_adv_stats("short_2", adv_short_2)
            if self.config_update_interval > 0 and (datetime.now() - self.last_config_update).total_seconds() > self.config_update_interval:
                self.update_normalization_config()
                self.last_config_update = datetime.now()

        # DEBUG: Log action distribution to verify models are outputting signals
        if self.config.get('runmode') in ['live', 'dry_run']:
            # Логируем Q-значения для последней свечи, чтобы видеть "уверенность" модели
            if "long_1" in q_values:
                a = action_long_1[-1]
                a_str = "HOLD" if a == 0 else ("ENTRY_LONG" if a == 1 else "OPPOSITE(SHORT)")
                adv = adv_long_1[-1]

                # Вычисляем РЕАЛЬНЫЙ порог голосования
                cfg = self.q_normalization.get("long_1", {})
                q_min = cfg.get('q_min', 0.0)
                q_max = cfg.get('q_max', q_min)
                thr = q_min + (q_max - q_min) * self.epsilon_threshold_eff_long if q_max > q_min else q_min

                norm = self._normalize_q_value(adv, "long_1")
                vote = adv > thr
                vote_mark = "VOTE" if vote else "NO"

                if adv > q_min:
                    self.logger.info(f"{metadata['pair']} L1: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

            if "long_2" in q_values:
                a = action_long_2[-1]
                a_str = "HOLD" if a == 0 else ("ENTRY_LONG" if a == 1 else "OPPOSITE(SHORT)")
                adv = adv_long_2[-1]

                cfg = self.q_normalization.get("long_2", {})
                q_min = cfg.get('q_min', 0.0)
                q_max = cfg.get('q_max', q_min)
                thr = q_min + (q_max - q_min) * self.epsilon_threshold_eff_long if q_max > q_min else q_min

                norm = self._normalize_q_value(adv, "long_2")
                vote = adv > thr
                vote_mark = "VOTE" if vote else "NO"

                if adv > q_min:
                    self.logger.info(f"{metadata['pair']} L2: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

            if "short_1" in q_values:
                a = action_short_1[-1]
                if self.short_1_is_mirror:
                    a_str = "HOLD" if a == 0 else ("ENTRY_SHORT" if a == 1 else "OPPOSITE(LONG)")
                else:
                    a_str = "HOLD" if a == 0 else ("LONG" if a == 1 else "ENTRY_SHORT")
                adv = adv_short_1[-1]

                cfg = self.q_normalization.get("short_1", {})
                q_min = cfg.get('q_min', 0.0)
                q_max = cfg.get('q_max', q_min)
                thr = q_min + (q_max - q_min) * self.epsilon_threshold_eff_short if q_max > q_min else q_min

                norm = self._normalize_q_value(adv, "short_1")
                vote = adv > thr
                vote_mark = "VOTE" if vote else "NO"

                if adv > q_min:
                    self.logger.info(f"{metadata['pair']} S1: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

            if "short_2" in q_values:
                a = action_short_2[-1]
                if self.short_2_is_mirror:
                    a_str = "HOLD" if a == 0 else ("ENTRY_SHORT" if a == 1 else "OPPOSITE(LONG)")
                else:
                    a_str = "HOLD" if a == 0 else ("LONG" if a == 1 else "ENTRY_SHORT")
                adv = adv_short_2[-1]

                cfg = self.q_normalization.get("short_2", {})
                q_min = cfg.get('q_min', 0.0)
                q_max = cfg.get('q_max', q_min)
                thr = q_min + (q_max - q_min) * self.epsilon_threshold_eff_short if q_max > q_min else q_min

                norm = self._normalize_q_value(adv, "short_2")
                vote = adv > thr
                vote_mark = "VOTE" if vote else "NO"

                if adv > q_min:
                    self.logger.info(f"{metadata['pair']} S2: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

        # 6. Применяем строгое голосование для каждой свечи
        n_predictions = len(action_long_1)
        target_idx = dataframe.index[-n_predictions:]
        
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        if 'exit_long' not in dataframe.columns:
            dataframe['exit_long'] = 0
        if 'exit_short' not in dataframe.columns:
            dataframe['exit_short'] = 0
        
        dataframe['enter_long'] = dataframe['enter_long'].astype(np.int8)
        dataframe['enter_short'] = dataframe['enter_short'].astype(np.int8)
        dataframe['exit_long'] = dataframe['exit_long'].astype(np.int8)
        dataframe['exit_short'] = dataframe['exit_short'].astype(np.int8)
        
        # Получаем информацию об открытой позиции по данному тикеру
        has_long = False
        has_short = False
        try:
            open_trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == metadata['pair']]).first()  # type: ignore
            has_long = open_trade.is_short is False if open_trade else False
            has_short = open_trade.is_short is True if open_trade else False
        except Exception:
            pass

        # --- OPTIMIZATION: Pre-allocate arrays to avoid dataframe.loc inside loop ---
        enter_long_vals = np.zeros(n_predictions, dtype=np.int8)
        enter_short_vals = np.zeros(n_predictions, dtype=np.int8)

        # --- Volume Filter Preparation ---
        volume_vals = None
        if 'quote_volume_sma' in dataframe.columns:
            min_volume = self.min_quote_volume_usd.value
            volume_vals = dataframe['quote_volume_sma'].iloc[-n_predictions:].values
            # Оптимизация для live/dry-run: если последняя свеча не проходит, выходим сразу
            if self.config.get('runmode') in ['live', 'dry_run']:
                if volume_vals[-1] < min_volume:
                    self.logger.info(
                        f"[LOW_VOLUME] {metadata['pair']}: Volume too low "
                        f"({volume_vals[-1]:.0f} < {min_volume:.0f} USD). "
                        f"Skipping signal generation."
                    )
                    # Убедимся, что колонки существуют, прежде чем возвращать
                    if 'enter_long' not in dataframe.columns: dataframe['enter_long'] = 0
                    if 'enter_short' not in dataframe.columns: dataframe['enter_short'] = 0
                    return dataframe

        # Prepare regime arrays for backtest filtering
        regime_global_vals = None
        if is_backtest and self.use_global_regime_filter and 'st_regime_global' in dataframe.columns:
            regime_global_vals = dataframe['st_regime_global'].iloc[-n_predictions:].values

        regime_local_vals = None
        if is_backtest and self.use_local_regime_filter and 'st_regime_local' in dataframe.columns:
            regime_local_vals = dataframe['st_regime_local'].iloc[-n_predictions:].values

        # --- VECTORIZED ENSEMBLE VOTING ---
        thresh_long = self.rl_long_threshold_opt.value
        thresh_short = self.rl_short_threshold_opt.value

        # 1. Create Vote Matrices (Boolean Arrays converted to Int)
        # Using arrays derived from get_action_with_threshold
        v_l1 = action_long_1.astype(np.int8) if self.enable_long_1 else np.zeros(n_predictions, dtype=np.int8)
        v_l2 = action_long_2.astype(np.int8) if self.enable_long_2 else np.zeros(n_predictions, dtype=np.int8)

        v_s1 = action_short_1.astype(np.int8) if self.enable_short_1 else np.zeros(n_predictions, dtype=np.int8)
        v_s2 = action_short_2.astype(np.int8) if self.enable_short_2 else np.zeros(n_predictions, dtype=np.int8)

        votes_long = v_l1 + v_l2
        votes_short = v_s1 + v_s2

        # Сохраняем голоса в датафрейм, чтобы они были доступны в custom_exit и других методах
        if is_backtest or n_predictions > 0:
            # Создаем пустые колонки, если их нет
            if 'votes_long' not in dataframe.columns:
                dataframe['votes_long'] = 0
            if 'votes_short' not in dataframe.columns:
                dataframe['votes_short'] = 0
            
            # Заполняем последние n_predictions строк (те, что предсказали модели)
            dataframe.iloc[-n_predictions:, dataframe.columns.get_loc('votes_long')] = votes_long
            dataframe.iloc[-n_predictions:, dataframe.columns.get_loc('votes_short')] = votes_short

        # 2. Veto Logic Arrays
        long_vetoed = (votes_short > 0) & self.enable_veto
        short_vetoed = (votes_long > 0) & self.enable_veto

        # 3. Final Signal Arrays (Fixed boolean logic)
        # Мы НЕ ограничиваем расчет сигналов текущей позицией, чтобы они были доступны для выхода (Alpha Decay)
        raw_enter_long = (votes_long >= thresh_long) & (~long_vetoed)
        raw_enter_short = (votes_short >= thresh_short) & (~short_vetoed) & self.can_short

        if is_backtest or n_predictions > 0:
            raw_enter_long &= dataframe['impulse_long_ok'].iloc[-n_predictions:].values
            raw_enter_short &= dataframe['impulse_short_ok'].iloc[-n_predictions:].values

        # Фильтр входов: для реального совершения сделки Freqtrade всё равно проверит наличие позиции,
        # но колонки enter_long/short теперь будут содержать сигналы всегда для работы логики выхода.

        # --- LOGGING RAW SIGNALS FOR SLIPPAGE ANALYTICS ---
        if not is_backtest:
            if isinstance(raw_enter_long, pd.Series):
                 if raw_enter_long.iloc[-1]:
                     self.logger.info(f"[RAW RL SIGNAL] {metadata['pair']} ENTRY LONG request generated.")
            elif isinstance(raw_enter_long, np.ndarray):
                 if raw_enter_long[-1]:
                     self.logger.info(f"[RAW RL SIGNAL] {metadata['pair']} ENTRY LONG request generated.")

            if isinstance(raw_enter_short, pd.Series):
                 if raw_enter_short.iloc[-1]:
                     self.logger.info(f"[RAW RL SIGNAL] {metadata['pair']} ENTRY SHORT request generated.")
            elif isinstance(raw_enter_short, np.ndarray):
                 if raw_enter_short[-1]:
                     self.logger.info(f"[RAW RL SIGNAL] {metadata['pair']} ENTRY SHORT request generated.")

        # --- APPLY VOLUME FILTERS (Gatekeepers) ---
        # Оптимизация: извлекаем только последние n_predictions строк, чтобы совпадала размерность
        if is_backtest or n_predictions > 0:
            df_tail = dataframe.iloc[-n_predictions:]

            # Retrieve hyperopt values
            f1_surge_val = float(self.vol_f1_surge.value)
            f1_pct_val = float(self.vol_f1_pct.value)
            f2_cvd_val = float(self.vol_f2_cvd_spike.value)
            f2_gap_val = float(self.vol_f2_gap.value)

            # Filter 1: Accum + Sweep
            f1_long = (df_tail['surge_ratio'].values > f1_surge_val) & (df_tail['buy_vol_pct'].values > f1_pct_val)
            f1_short = (df_tail['surge_ratio'].values > f1_surge_val) & (df_tail['sell_vol_pct'].values > f1_pct_val)

            # Filter 2: Imbalance (CVD Spike & Gap Fill)
            cvd_v = df_tail['cvd'].values
            cvd_ma_v = df_tail['cvd_ma'].values

            f2_long = (cvd_v > f2_cvd_val * cvd_ma_v) & (df_tail['gap_pct_long'].values > f2_gap_val)
            f2_short = (cvd_v < -f2_cvd_val * cvd_ma_v) & (df_tail['gap_pct_short'].values > f2_gap_val)

            # Intersection: RL Signal + Filter 1 + Filter 2
            # Если фильтры слишком жесткие для текущей фазы тестов, можно закомментировать f2
            
            # Apply Filter 1 (Surge) if enabled
            if self.vol_f1_enabled.value:
                raw_enter_long = raw_enter_long & f1_long
                raw_enter_short = raw_enter_short & f1_short
            
            # Apply Filter 2 (CVD) if enabled
            if self.vol_f2_enabled.value:
                raw_enter_long = raw_enter_long & f2_long
                raw_enter_short = raw_enter_short & f2_short

            # --- LOGGING FILTERED SIGNALS ---
            if not is_backtest:
                if isinstance(raw_enter_long, pd.Series):
                     if raw_enter_long.iloc[-1]:
                         self.logger.info(f"[FILTERED SIGNAL] {metadata['pair']} ENTRY LONG passed volume filters.")
                elif isinstance(raw_enter_long, np.ndarray):
                     if raw_enter_long[-1]:
                         self.logger.info(f"[FILTERED SIGNAL] {metadata['pair']} ENTRY LONG passed volume filters.")

                if isinstance(raw_enter_short, pd.Series):
                     if raw_enter_short.iloc[-1]:
                         self.logger.info(f"[FILTERED SIGNAL] {metadata['pair']} ENTRY SHORT passed volume filters.")
                elif isinstance(raw_enter_short, np.ndarray):
                     if raw_enter_short[-1]:
                         self.logger.info(f"[FILTERED SIGNAL] {metadata['pair']} ENTRY SHORT passed volume filters.")

        enter_long_vals = raw_enter_long.astype(np.int8)
        enter_short_vals = raw_enter_short.astype(np.int8)

        # 4. Apply Volume & Regime Filters Vectorized (for Backtest)
        if volume_vals is not None:
            # Возвращаемся к значению из конфига, не душим модель
            vol_mask = volume_vals >= self.min_quote_volume_usd.value
            
            enter_long_vals &= vol_mask
            enter_short_vals &= vol_mask
            
        if is_backtest:
            if self.use_global_regime_filter and regime_global_vals is not None:
                enter_long_vals &= (regime_global_vals > 0)
                enter_short_vals &= (regime_global_vals < 0)
            if self.use_local_regime_filter and regime_local_vals is not None:
                enter_long_vals &= (regime_local_vals > 0)
                enter_short_vals &= (regime_local_vals < 0)
        else:
            # Live mode strict flags
            if not allow_long: enter_long_vals.fill(0)
            if not allow_short: enter_short_vals.fill(0)
            
            # Log only the last signal in live mode if generated
            if enter_long_vals[-1] == 1:
                self.logger.info(f"[SIGNAL] {metadata['pair']} ENTRY LONG Vectorized logic triggered.")
            elif enter_short_vals[-1] == 1:
                self.logger.info(f"[SIGNAL] {metadata['pair']} ENTRY SHORT Vectorized logic triggered.")

        # Assign back to DataFrame
        dataframe.loc[target_idx, 'enter_long'] = enter_long_vals
        dataframe.loc[target_idx, 'enter_short'] = enter_short_vals
        dataframe.loc[target_idx, 'votes_long'] = votes_long
        dataframe.loc[target_idx, 'votes_short'] = votes_short

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Логика выходов: 
        1. Фильтр 3: Climax reversal (Vol peak + divergence).
        (Alpha Decay перенесен в custom_exit, чтобы работать только при убытках)
        """
        if not self.vol_f3_enabled.value:
            return dataframe

        if dataframe.empty or 'surge_ratio' not in dataframe.columns:
            return dataframe

        # Divergence proxy: peak ratio > 5, but directional volume drops below 30% of recent maximum
        buy_peak_rolling = dataframe['buy_vol_pct'].rolling(window=10).max()
        sell_peak_rolling = dataframe['sell_vol_pct'].rolling(window=10).max()

        f3_peak_val = float(self.vol_f3_peak.value)
        f3_fade_val = float(self.vol_f3_fade.value)

        # Vectorized conditions
        exit_long_cond = (dataframe['surge_ratio'] > f3_peak_val) & (dataframe['buy_vol_pct'] < (buy_peak_rolling * f3_fade_val))
        exit_short_cond = (dataframe['surge_ratio'] > f3_peak_val) & (dataframe['sell_vol_pct'] < (sell_peak_rolling * f3_fade_val))

        # Merge with existing exits if any (using np.where for safe int8 casting)
        dataframe['exit_long'] = np.where(exit_long_cond, 1, dataframe.get('exit_long', 0))
        dataframe['exit_short'] = np.where(exit_short_cond, 1, dataframe.get('exit_short', 0))

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Обязательный метод для торговли фьючерсами.
        В логах видно, что proposed_leverage может приходить как 1.0, даже если в конфиге указано иное.
        Этот код добавляет "защиту", чтобы всегда использовать плечо из конфига.
        """
        leverage_conf = self.config.get('leverage', {})

        # Ищем плечо для конкретной пары
        if pair in leverage_conf:
            return float(leverage_conf[pair])

        # Если для пары нет, ищем значение по умолчанию "*"
        if '*' in leverage_conf:
            return float(leverage_conf['*'])

        # Если ничего не найдено, возвращаем предложенное значение (вероятно, 1.0)
        self.logger.warning(f"Leverage not found for {pair} in config. Falling back to proposed: {proposed_leverage}")
        return proposed_leverage