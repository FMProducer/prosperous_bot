import sys
import logging
import logging.handlers
import importlib.util
from pathlib import Path
import json
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas import DataFrame  # type: ignore
import torch  # type: ignore
from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Optional, List, Any, Tuple
from collections import deque


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
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair  # type: ignore
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
    startup_candle_count: int = 200

    # Информативный таймфрейм для режима рынка
    informative_timeframe = '15m'
    informative_timeframe_global = '15m'
    
    minimal_roi = {"0": 100}
    stoploss = -0.99  # Заглушка, работает custom_stoploss
    trailing_stop = False
    use_custom_stoploss = True
    
    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # Параметры TSL
    # load=False и optimize=False гарантируют использование значений default (из обучения)
    d0 = DecimalParameter(0.01, 0.10, default=0.0752, space='sell', optimize=True, load=True)
    d_min = DecimalParameter(0.0005, 0.05, default=0.00082, space='sell', optimize=True, load=True)
    hysteresis = DecimalParameter(0.00005, 0.01, default=0.00005, space='sell', optimize=True, load=True)
    # Расширим диапазон p_target, чтобы Hyperopt мог проверить и более консервативные (до 10%), и агрессивные варианты
    p_target = DecimalParameter(0.01, 0.10, default=0.0217, space='sell', optimize=True, load=True)
    
    # Hyperoptable Voting Thresholds
    rl_long_threshold_opt = IntParameter(1, 2, default=1, space='buy', optimize=True, load=True)
    rl_short_threshold_opt = IntParameter(1, 2, default=1, space='sell', optimize=True, load=True)

    # Параметры Supertrend для режима рынка (будут оптимизироваться hyperopt'ом)
    supertrend_period = IntParameter(7, 20, default=10, space='buy', optimize=True, load=True)
    supertrend_multiplier = DecimalParameter(1.5, 4.0, default=3.0, space='buy', optimize=True, load=True)

    # Фильтр по объему для отсеивания неликвидных пар (особенно в бэктесте)
    min_quote_volume_usd = DecimalParameter(0, 500000, default=100000, space='buy', optimize=True, load=True)

    # Коэффициент агрессии для Dynamic Epsilon (чувствительность к просадке)
    dd_aggression_k = DecimalParameter(0.1, 2.0, default=0.55, space='buy', optimize=True, load=True)

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
        
        # Принудительно включаем шорты
        self.can_short = True
        
        # --- PERFORMANCE OPTIMIZATION: GLOBAL REGIME CACHE ---
        # Кэш для хранения результатов расчета индикаторов на информативных таймфреймах (BTC 1h)
        # Ключ: (timeframe, last_candle_timestamp), Значение: готовый DataFrame с индикаторами
        self._global_regime_cache = {}
        self._global_regime_lock = threading.Lock()
        
        # Override min_quote_volume_usd from config if present
        if 'min_quote_volume_usd' in config:
            self.min_quote_volume_usd.value = float(config['min_quote_volume_usd'])
            logger.info(f"[CONFIG] min_quote_volume_usd overridden from config: {self.min_quote_volume_usd.value}")
        
        # Загрузка dd_aggression_k из конфига, если он там есть
        if 'dd_aggression_k' in config.get('rl_ensemble', {}):
            self.dd_aggression_k.value = float(config['rl_ensemble']['dd_aggression_k'])
            logger.info(f"[CONFIG] dd_aggression_k overridden from config: {self.dd_aggression_k.value}")

        # --- LOGGING FILTERS ---
        # Убираем спам о отмене стоплосса
        def filter_stoploss_cancel(record):
            msg = record.getMessage()
            return "Cancelling stoploss on exchange" not in msg and "Cancelling current stoploss on exchange" not in msg
        logging.getLogger('freqtrade.freqtradebot').addFilter(filter_stoploss_cancel)
        
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
        # 2. ThreadPool для параллельного inference
        self.executor = ThreadPoolExecutor(max_workers=num_cpu_threads)
        
        # 3. Кэш для feature tensors (экономим на preprocessing)
        self.feature_cache: Dict[tuple, Any] = {}
        self.q_value_cache: Dict[tuple, Dict[str, np.ndarray]] = {}  # Кэш для результатов инференса (Q-values)
        self.cache_lock = threading.Lock()
        self.cache_max_size = 100  # храним только последние 100 пар свечей
        
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

        logger.info(
            f"🎰 Dynamic Slots: {'ENABLED' if self.dynamic_slots_enabled else 'DISABLED'} "
            f"| Total: {self.total_slots}"
        )

        # --- ПУТИ К 4 МОДЕЛЯМ ---
        # Long Model 1:
        self.long_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260125_time_033653"
        self.long_1_model_pth = self.long_1_model_dir / "best.pth"
        
        # Long Model 2:
        self.long_2_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260201_time_131607_no tsl"
        self.long_2_model_pth = self.long_2_model_dir / "best.pth"
        pth = self.long_2_model_dir / "best.pth"
        
        # Short Model 1:
        self.short_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260213_time_004217"
        self.short_1_model_pth = self.short_1_model_dir / "best.pth"
        
        # Short Model 2:
        self.short_2_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260212_time_020927"
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

        # Включение/выключение regime-фильтра (Supertrend на 15m)
        self.use_global_regime_filter: bool = self.ensemble_cfg.get('use_global_regime_filter', True)
        self.use_local_regime_filter: bool = self.ensemble_cfg.get('use_local_regime_filter', True)

        # --- DYNAMIC EPSILON V2 CONFIG ---
        # Base epsilon from config (common legacy value)
        self.epsilon_threshold: float = self.ensemble_cfg.get('epsilon_threshold', 0.75) # Legacy fallback

        # V2: Base/Target Epsilon
        self.epsilon_threshold_long: float = self.ensemble_cfg.get("epsilon_threshold_long", 0.75)
        self.epsilon_threshold_short: float = self.ensemble_cfg.get("epsilon_threshold_short", 0.75)

        # V2: Epsilon Range
        self.epsilon_min: float = self.ensemble_cfg.get("epsilon_min", 0.1)
        self.epsilon_max: float = self.ensemble_cfg.get("epsilon_max", 1.0)

        # Effective epsilons used for thresholding, updated dynamically
        self.epsilon_threshold_eff_long: float = float(self.epsilon_threshold_long)
        self.epsilon_threshold_eff_short: float = float(self.epsilon_threshold_short)
        # Aggregate value for logging/compatibility
        self.epsilon_threshold_eff: float = 0.5 * (
            self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short
        )

        # V2: Максимальная наблюдаемая нереализованная прибыль (PnL) по направлениям
        self.pnl_max_long: float = 0.0
        self.pnl_max_short: float = 0.0

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
            self.epsilon_threshold_eff = 0.5 * (self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short)
            
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
            f"Local Filter (Pair 15m): {'ON' if self.use_local_regime_filter else 'OFF'}"
        )

        self.logger.info(
            f"[ENSEMBLE] Ensemble Config: EpsL={self.epsilon_threshold_long} | EpsS={self.epsilon_threshold_short} | "
            f"Range=[{self.epsilon_min}, {self.epsilon_max}]"
        )

        # --- ИНИЦИАЛИЗАЦИЯ 4 АГЕНТОВ ---
        logger.info("[AGENTS] Creating agents...")
        
        # Long 1
        if self.enable_long_1:
            self.long_1_agent = self._create_agent_from_config(self.cfg_long_1, mirror_mode=False)
            self._load_weights(self.long_1_agent, self.long_1_model_pth, "LONG_1")
        else:
            self.long_1_agent = None
            
        # Long 2
        if self.enable_long_2:
            self.long_2_agent = self._create_agent_from_config(self.cfg_long_2, mirror_mode=False)
            self._load_weights(self.long_2_agent, self.long_2_model_pth, "LONG_2")
        else:
            self.long_2_agent = None
            
        # Short 1
        if self.enable_short_1:
            self.short_1_agent = self._create_agent_from_config(self.cfg_short_1, mirror_mode=self.short_1_is_mirror)
            self._load_weights(self.short_1_agent, self.short_1_model_pth, "SHORT_1")
        else:
            self.short_1_agent = None
            
        # Short 2
        if self.enable_short_2:
            self.short_2_agent = self._create_agent_from_config(self.cfg_short_2, mirror_mode=self.short_2_is_mirror)
            self._load_weights(self.short_2_agent, self.short_2_model_pth, "SHORT_2")
        else:
            self.short_2_agent = None
        
        # --- ЗАГРУЗКА ВЕСОВ ---
        # Safety check: Ensure Long and Short models are not pointing to the same file
        if self.long_1_model_pth == self.short_1_model_pth:
            logger.error("🚨 CRITICAL: LONG_1 and SHORT_1 model paths are IDENTICAL! Check paths.")
        if self.long_2_model_pth == self.short_2_model_pth:
            logger.error("🚨 CRITICAL: LONG_2 and SHORT_2 model paths are IDENTICAL! Check paths.")

        # === ОПТИМИЗАЦИЯ МОДЕЛЕЙ ДЛЯ INFERENCE ===
        # После загрузки весов, оптимизируем модели
        logger.info("[INFO] Optimizing models for CPU inference...")
        
        # Переводим в eval mode и оптимизируем
        agents_to_optimize = []
        if self.enable_long_1: agents_to_optimize.append(("LONG_1", self.long_1_agent))
        if self.enable_long_2: agents_to_optimize.append(("LONG_2", self.long_2_agent))
        if self.enable_short_1: agents_to_optimize.append(("SHORT_1", self.short_1_agent))
        if self.enable_short_2: agents_to_optimize.append(("SHORT_2", self.short_2_agent))

        for agent_name, agent in agents_to_optimize:
            if agent:
                agent.policy_net.eval()
                
                # Отключаем grad для всех параметров (экономит память и время)
                for param in agent.policy_net.parameters():
                    param.requires_grad = False
        
        logger.info("[OK] CPU optimizations applied")
        
        if not self.can_short:
            logger.warning("⚠️ WARNING: can_short is False! Short signals will be ignored.")

        logger.info("=" * 60)
        logger.info("✅ 2+2 ENSEMBLE READY FOR TRADING")
        logger.info("=" * 60)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('executor', None)
        state.pop('cache_lock', None)
        state.pop('logger', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = logging.getLogger(__name__)
        self.cache_lock = threading.Lock()
        num_cpu_threads = self.config.get('cpu_threads', 4)
        self.executor = ThreadPoolExecutor(max_workers=num_cpu_threads)

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
    
    def _load_weights(self, agent, path, name):
        try:
            agent.load_model(str(path))
            agent.policy_net.eval()
            logger.info(f"✅ {name} Agent loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name} Agent: {e}")
            raise e

    # === HELPER: Supertrend на одном таймфрейме (для 15m режима) ===
    def _compute_supertrend(self, df: DataFrame, period: int, multiplier: float) -> DataFrame:
        """
        Вычисляет Supertrend для данного OHLCV DataFrame.
        Оптимизировано с использованием NumPy для ускорения расчетов.
        """
        if df is None or df.empty:
            return pd.DataFrame(index=df.index if df is not None else None)

        # 1. Расчет ATR (остаемся в Pandas, т.к. ewm оптимизирован)
        high = df['high']
        low = df['low']
        close = df['close']
        
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()

        # 2. Подготовка базовых линий
        mid = (high + low) / 2.0
        upperband_p = (mid + multiplier * atr).ffill().values
        lowerband_p = (mid - multiplier * atr).ffill().values
        close_p = close.values
        
        # 3. Основной цикл на NumPy (убираем .iloc, который сильно тормозит)
        size = len(df)
        st = np.zeros(size, dtype=np.float64)
        direction = np.ones(size, dtype=np.int8)
        
        # Начальные значения
        st[0] = upperband_p[0]
        direction[0] = 1
        
        for i in range(1, size):
            # Предварительное направление на основе предыдущей ленты
            if close_p[i] > upperband_p[i - 1]:
                direction[i] = 1
            elif close_p[i] < lowerband_p[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
                
                # Трейлинг лент (самая тяжелая логика ST)
                if direction[i] == 1:
                    if upperband_p[i] > upperband_p[i - 1]:
                        upperband_p[i] = upperband_p[i - 1]
                else:
                    if lowerband_p[i] < lowerband_p[i - 1]:
                        lowerband_p[i] = lowerband_p[i - 1]
            
            # Результирующее значение ST
            st[i] = lowerband_p[i] if direction[i] == 1 else upperband_p[i]

        out = pd.DataFrame(index=df.index)
        out['st'] = st
        out['st_dir'] = direction
        return out

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Z-score нормализация (как в обучении)
        zscore_window = 450
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            rolling = dataframe[col].rolling(window=zscore_window, min_periods=zscore_window)
            mean = rolling.mean()
            std = rolling.std(ddof=0)
            # Z-score: (x - mean) / std
            dataframe[f'{col}_z'] = (dataframe[col] - mean) / (std + 1e-8)

        # Заполняем NaN нулями (начало датафрейма), чтобы модель не получала inf/nan
        z_cols = [f'{col}_z' for col in ohlcv_cols]
        dataframe[z_cols] = dataframe[z_cols].fillna(0.0)

        # 2. Quote Volume для фильтрации неликвида
        # Расчет среднего объема за 24 часа (1440 свечей на 1m)
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['close']
        dataframe['quote_volume_sma'] = dataframe['quote_volume'].rolling(window=1440, min_periods=200).mean()
        dataframe['quote_volume_sma'] = dataframe['quote_volume_sma'].fillna(0)

        # 2. Supertrend Режимы
        if hasattr(self, 'dp') and getattr(self, 'dp', None) is not None:
            st_period = int(self.supertrend_period.value)
            st_mult = float(self.supertrend_multiplier.value)

            # --- GLOBAL REGIME (BTC 15m) ---
            # Calculate always for dashboard visibility, even if filter is disabled
            if True:
                try:
                    inf_tf_global = self.informative_timeframe_global
                    btc_df = self.dp.get_pair_dataframe('BTC/USDT:USDT', inf_tf_global)
                    if btc_df is not None and not btc_df.empty:
                        last_ts = str(btc_df['date'].iloc[-1])
                        # Уникальный ключ кэша для глобального режима
                        cache_key = ("GLOBAL", "BTC/USDT:USDT", inf_tf_global, last_ts)
                        
                        merge_df = None
                        with self._global_regime_lock:
                            if cache_key in self._global_regime_cache:
                                merge_df = self._global_regime_cache[cache_key]
                        
                        if merge_df is None:
                            # 1. Сначала вычисляем SuperTrend (Тяжелая операция)
                            st_global = self._compute_supertrend(
                                btc_df[['high', 'low', 'close']],
                                period=st_period,
                                multiplier=st_mult,
                            )
                            if not st_global.empty:
                                # 2. Создаем временный DF для мерджа
                                merge_df = btc_df[['date']].copy()
                                merge_df['st_regime_global'] = st_global['st_dir'].values
                                
                                # Сохраняем в кэш
                                with self._global_regime_lock:
                                    # Очистка старого кэша (чтобы не рос)
                                    if len(self._global_regime_cache) > 200:
                                        self._global_regime_cache.clear()
                                    self._global_regime_cache[cache_key] = merge_df

                        if merge_df is not None:
                            # 3. Динамическая подстройка TZ (ВАЖНО для предотвращения падения)
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
                            
                            # Извлекаем из колонки с суффиксом (напр. st_regime_global_1h)
                            inf_col = f"st_regime_global_{inf_tf_global}"
                            if inf_col in dataframe.columns:
                                dataframe['st_regime_global'] = dataframe[inf_col].fillna(0).astype(np.int8)
                except Exception as e:
                    self.logger.warning(f"Global regime calculation failed: {e}")

            # --- LOCAL REGIME (Current Pair 15m) ---
            if self.use_local_regime_filter:
                try:
                    inf_tf_local = self.informative_timeframe
                    local_df = self.dp.get_pair_dataframe(metadata['pair'], inf_tf_local)
                    if local_df is not None and not local_df.empty:
                        last_ts_local = str(local_df['date'].iloc[-1])
                        # Кэшируем и локальный режим (бывает полезно при параллельном вызове)
                        cache_key_local = ("LOCAL", metadata['pair'], inf_tf_local, last_ts_local)
                        
                        merge_df_local = None
                        with self._global_regime_lock:
                            if cache_key_local in self._global_regime_cache:
                                merge_df_local = self._global_regime_cache[cache_key_local]
                                
                        if merge_df_local is None:
                            st_local = self._compute_supertrend(
                                local_df[['high', 'low', 'close']],
                                period=st_period,
                                multiplier=st_mult,
                            )
                            if not st_local.empty:
                                # 2. Создаем временный DF
                                merge_df_local = local_df[['date']].copy()
                                merge_df_local['st_regime_local'] = st_local['st_dir'].values
                                
                                with self._global_regime_lock:
                                    self._global_regime_cache[cache_key_local] = merge_df_local

                        if merge_df_local is not None:
                            # 3. Динамическая подстройка TZ
                            main_tz = dataframe['date'].dt.tz
                            if merge_df_local['date'].dt.tz != main_tz:
                                merge_df_local = merge_df_local.copy()
                                if main_tz is None:
                                    merge_df_local['date'] = merge_df_local['date'].dt.tz_localize(None)
                                else:
                                    if merge_df_local['date'].dt.tz is None:
                                        merge_df_local['date'] = merge_df_local['date'].dt.tz_localize(main_tz)
                                    else:
                                        merge_df_local['date'] = merge_df_local['date'].dt.tz_convert(main_tz)
                            
                            dataframe = merge_informative_pair(
                                dataframe, merge_df_local, 
                                self.timeframe, inf_tf_local, ffill=True
                            )
                            
                            # Извлекаем из колонки с суффиксом (напр. st_regime_local_15m)
                            inf_col_local = f"st_regime_local_{inf_tf_local}"
                            if inf_col_local in dataframe.columns:
                                dataframe['st_regime_local'] = dataframe[inf_col_local].fillna(0).astype(np.int8)
                except Exception as e:
                    self.logger.warning(f"Local regime calculation failed for {metadata['pair']}: {e}")

        return dataframe

    def informative_pairs(self):
        """
        Информативные пары:
        1. Все пары из whitelist на таймфрейме 15m (локальный режим)
        2. BTC/USDT на таймфрейме 1h (глобальный режим)
        """
        pairs: List[Any] = []
        if hasattr(self, 'dp') and getattr(self, 'dp', None) is not None:
            try:
                pairs = self.dp.current_whitelist()  # type: ignore
            except Exception:
                pairs = []
        
        if not pairs:
            pairs = self.config.get('exchange', {}).get('pair_whitelist', [])

        # Формируем список: (pair, 15m) для всех
        info_list = [(pair, self.informative_timeframe) for pair in pairs]
        
        # Добавляем принудительно BTC на 15m для глобального режима
        btc_global = ('BTC/USDT:USDT', self.informative_timeframe_global)
        if btc_global not in info_list:
            info_list.append(btc_global)
            
        return info_list
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs):
        trade_open_date = getattr(trade, 'open_date_utc', None)
        if trade_open_date is not None:
            duration_min = (current_time - trade_open_date).total_seconds() / 60
            if duration_min >= 60:
                return "timeout_60m"
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

        # НЕЛИНЕЙНЫЙ ТРЕЙЛИНГ (агрессивный в начале)
        # Стоп подтягивается быстро при малом профите (корень квадратный)
        if calc_p <= FEE_BUF:
            d_eff = d0_val
        else:
            p_factor = calc_p - FEE_BUF
            # Целевой профит, при котором отступ сужается до d_min (берем из параметра)
            p_target_val = self.p_target.value
            # Нормализуем (0..1)
            p_norm = min(1.0, p_factor / p_target_val) 
            # Степенная функция (0.5 = корень). Чем меньше степень, тем резче старт.
            d_eff = d0_val - (d0_val - d_min_val) * (p_norm**0.3)
            d_eff = max(d_min_val, d_eff)
        
        return -d_eff

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str) -> Optional[torch.Tensor]:
        # Default window is 90 (5 channels * 90 = 450 history features)
        window = 90 
        
        # Safe access to model configs to avoid Pyre NoneType errors
        cfg = None
        if side == "LONG":
            cfg = self.cfg_long_1 if model_num == 1 else self.cfg_long_2
        elif side == "SHORT":
            cfg = self.cfg_short_1 if model_num == 1 else self.cfg_short_2
            
        if cfg and hasattr(cfg, 'seq') and cfg.seq:
            window = getattr(cfg.seq, "agent_history_len", 90)

        # 1. Проверка длины
        if len(dataframe) < window:
            return None
            
        # 2. Выбор Z-колонок
        cols = ['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']
        
        # 3. Data preparation (N, 5)
        z_data = dataframe[cols].values.astype(np.float32)
        
        # Extract last 'window' rows: (window, 5)
        # Using tail instead of sliding_window_view for simplicity since we only need 1 window
        last_window = z_data[-window:]
        
        # 4. Inversion logic (Mirror Mode)
        should_invert = False
        if side == "SHORT":
            if (model_num == 1 and self.short_1_is_mirror) or (model_num == 2 and self.short_2_is_mirror):
                should_invert = True
        
        if should_invert:
            # Создаем копию, чтобы не мутировать исходный массив
            inverted_window = last_window.copy()

            # 1. Инвертируем только ценовые каналы (Open, High, Low, Close -> индексы 0, 1, 2, 3)
            inverted_window[:, 0:4] = inverted_window[:, 0:4] * -1.0

            # 2. Делаем Swap для High и Low (индексы 1 и 2)
            # После умножения на -1, бывший Low стал самым большим (новым High),
            # а бывший High стал самым маленьким (новым Low).
            temp_new_high = inverted_window[:, 2].copy() # -old_low
            temp_new_low = inverted_window[:, 1].copy()  # -old_high

            inverted_window[:, 1] = temp_new_high
            inverted_window[:, 2] = temp_new_low

            # ВАЖНО: Канал 4 (volume_z) остается без изменений (inverted_window[:, 4] == last_window[:, 4])
            # Всплеск объемов должен оставаться всплеском.

            last_window = inverted_window

        # 5. Flatten to (450,) and concatenate with dummy additional features (4,)
        # The model expects a flat vector of size (channels * history_len + additional_feats)
        # Target shape for history: (channels=5, length=90) -> (450,)
        
        # Transpose (90, 5) -> (5, 90) then flatten
        img_flat = last_window.T.flatten()
        
        # Additional features (4 dummy values)
        add_feats = np.zeros(4, dtype=np.float32)
        
        # Concat: (454,)
        full_input = np.concatenate([img_flat, add_feats])
        
        # Convert to tensor and add batch dim: (1, 454)
        return torch.as_tensor(full_input, device=self.device, dtype=torch.float32).unsqueeze(0)

    def get_model_input_cached(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str):
        """
        Кэширующая версия get_model_input
        Ключ кэша = (pair, last_candle_timestamp, side, model_num)
        """
        # Генерируем ключ кэша
        last_timestamp = dataframe.iloc[-1]['date'] if 'date' in dataframe.columns else dataframe.index[-1]
        cache_key = (pair, str(last_timestamp), side, model_num)
        
        with self.cache_lock:
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
        
        tensor = self.get_model_input(dataframe, pair, side, model_num, asset_name)
        
        with self.cache_lock:
            if len(self.feature_cache) >= self.cache_max_size:
                self.feature_cache.pop(next(iter(self.feature_cache)))
            self.feature_cache[cache_key] = tensor
        
        return tensor

    def _parallel_inference(self, tensors_and_agents):
        def single_inference(tensor_or_tuple, agent):
            try:
                with torch.no_grad():
                    if isinstance(tensor_or_tuple, tuple):
                        return agent.policy_net(*tensor_or_tuple).cpu().numpy()
                    else:
                        return agent.policy_net(tensor_or_tuple).cpu().numpy()
            except Exception as e:
                # Determine batch size for the zero-array fallback
                batch_size = 1
                if isinstance(tensor_or_tuple, tuple) and len(tensor_or_tuple) > 0 and hasattr(tensor_or_tuple[0], 'shape'):
                    batch_size = tensor_or_tuple[0].shape[0]
                    img_shape = tensor_or_tuple[0].shape
                    feat_shape = tensor_or_tuple[1].shape if len(tensor_or_tuple) > 1 else 'N/A'
                    logger.error(f"Inference failed for shapes img={img_shape}, feat={feat_shape}: {e}")
                elif hasattr(tensor_or_tuple, 'shape'):
                    batch_size = tensor_or_tuple.shape[0]
                    logger.error(f"Inference failed for shape {tensor_or_tuple.shape}: {e}")
                else:
                    logger.error(f"Inference failed with unknown input type: {e}")

                return np.zeros((batch_size, agent.action_dim))
        
        futures = []
        for tensor, agent, name in tensors_and_agents:
            future = self.executor.submit(single_inference, tensor, agent)  # type: ignore
            futures.append((future, name))
        
        results = {}
        for future, name in futures:
            results[name] = future.result()
        
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
            self.adv_history[name] = deque(maxlen=5400)

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

    def _get_pnl_from_freqtrade(self) -> tuple:
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
                        if hasattr(self, 'dp') and getattr(self, 'dp', None) is not None:
                            try:
                                dataframe, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)  # type: ignore
                                if not dataframe.empty:
                                    current_rate = dataframe['close'].iloc[-1]
                                    profit_usdt = t.calc_profit(rate=current_rate)
                                else:
                                    profit_usdt = t.calc_profit(rate=t.open_rate)
                            except Exception:
                                profit_usdt = t.calc_profit(rate=t.open_rate)
                        else:
                            profit_usdt = t.calc_profit(rate=t.open_rate)
                except Exception as e:
                    logger.debug(f"Failed to calc profit for {t.pair}: {e}")
                    profit_usdt = 0.0

                if t.is_short:
                    pnl_short_open += profit_usdt  # type: ignore
                else:
                    pnl_long_open += profit_usdt  # type: ignore

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

    def _update_dynamic_epsilon(self) -> None:
        """
        Update effective epsilon based on current PnL drawdown per side (v2).
        In calibration mode, dynamic epsilon is disabled.
        """
        if getattr(self, "calibration_mode", False) or self.config.get("runmode") not in ("live", "dry_run"):
            # In calibration or backtest, use static base values
            self.epsilon_threshold_eff_long = float(self.epsilon_threshold_long)
            self.epsilon_threshold_eff_short = float(self.epsilon_threshold_short)
            self.epsilon_threshold_eff = 0.5 * (self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short)
            return

        try:
            # 1. Get current PnL and open trade counts per side
            pnl_long, pnl_short = self._get_pnl_from_freqtrade()
            open_trades = Trade.get_open_trades()
            count_long = sum(1 for t in open_trades if not t.is_short)
            count_short = sum(1 for t in open_trades if t.is_short)

            # 2. Independent reset of high-water mark
            if count_long == 0: self.pnl_max_long = 0.0
            if count_short == 0: self.pnl_max_short = 0.0

            # 3. Update high-water mark (peak PnL)
            if count_long > 0: self.pnl_max_long = max(self.pnl_max_long, pnl_long)
            if count_short > 0: self.pnl_max_short = max(self.pnl_max_short, pnl_short)

            # 4. Calculate drawdown relative to peak PnL (with protection for immediate drawdowns)
            # Using 10.0 as a base USDT denominator for immediate drops when max_pnl is 0
            min_denom = 10.0
            dd_long = (self.pnl_max_long - pnl_long) / max(self.pnl_max_long, min_denom)
            dd_short = (self.pnl_max_short - pnl_short) / max(self.pnl_max_short, min_denom)
            
            dd_long = max(0.0, min(dd_long, 1.0))
            dd_short = max(0.0, min(dd_short, 1.0))

            # 5. Calculate target Epsilon based on drawdown (Linear interpolation to max)
            k = self.dd_aggression_k.value
            # Map drawdown [0, 1] -> [epsilon_threshold, epsilon_max]
            target_eps_long = self.epsilon_threshold_long + (self.epsilon_max - self.epsilon_threshold_long) * min(dd_long * k, 1.0)
            target_eps_short = self.epsilon_threshold_short + (self.epsilon_max - self.epsilon_threshold_short) * min(dd_short * k, 1.0)

            # 6. Smooth update via EMA
            alpha = 0.3  # Smoothing factor
            self.epsilon_threshold_eff_long = (1.0 - alpha) * self.epsilon_threshold_eff_long + alpha * target_eps_long
            self.epsilon_threshold_eff_short = (1.0 - alpha) * self.epsilon_threshold_eff_short + alpha * target_eps_short

            # 7. Clamp effective thresholds into the configured [min, max] range
            self.epsilon_threshold_eff_long = float(min(max(self.epsilon_threshold_eff_long, self.epsilon_min), self.epsilon_max))
            self.epsilon_threshold_eff_short = float(min(max(self.epsilon_threshold_eff_short, self.epsilon_min), self.epsilon_max))

            # Update aggregate value for logging
            self.epsilon_threshold_eff = 0.5 * (self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short)

            # Logging for visibility
            if self.config.get("runmode") in ("live", "dry_run"):
                self.logger.debug(
                    f"EPS-DDv2 | ddL={dd_long:.2f} ddS={dd_short:.2f} | "
                    f"EpsL: {self.epsilon_threshold_long:.3f} -> {self.epsilon_threshold_eff_long:.3f} | "
                    f"EpsS: {self.epsilon_threshold_short:.3f} -> {self.epsilon_threshold_eff_short:.3f}"
                )
        except Exception as e:
            self.logger.warning(f"Dynamic epsilon v2 update failed: {e}. Falling back to base epsilon.")
            self.epsilon_threshold_eff_long = self.epsilon_threshold_long
            self.epsilon_threshold_eff_short = self.epsilon_threshold_short
            self.epsilon_threshold_eff = 0.5 * (self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short)

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
        # Determine thresholds (Hyperopt support)
        if self.config.get('runmode') == 'hyperopt':
            thresh_long = self.rl_long_threshold_opt.value
            thresh_short = self.rl_short_threshold_opt.value
        else:
            thresh_long = self.rl_long_threshold
            thresh_short = self.rl_short_threshold

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

            if adv <= q_min:
                return 0, True, 0.0

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
            from freqtrade.persistence import Trade  # type: ignore
            
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
                # Smart Cooldown: Блокируем только если сделка была убыточной
                if minutes_since < 90 and (last_trade.close_profit is not None and last_trade.close_profit < 0):
                    last_side = "short" if last_trade.is_short else "long"
                    # Блокируем только если направление совпадает (Long после Long или Short после Short)
                    if last_side == side:
                        self.logger.info(f"⏳ TIMEOUT {pair}: Last {last_side} (P={last_trade.close_profit:.2%}) closed {minutes_since:.1f}m ago. Blocking new {side}.")
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
    
    def _extract_features_batch(self, dataframe: DataFrame, window: int = 90, should_invert: bool = False) -> Optional[np.ndarray]:
        """
        Efficiently extract features for a single pair to be part of a batch.
        Optimized to minimize memory allocations.
        """
        if len(dataframe) < window:
            return None

        cols = ['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']
        try:
            # Get a view of the last N rows
            data_view = dataframe[cols].values[-window:]

            # Pre-allocate the resulting array to avoid np.concatenate
            # History is 5 channels * 90 window = 450 + 4 dummy features = 454
            feat = np.zeros(454, dtype=np.float32)

            # Efficiently fill the pre-allocated array
            # Transpose (window, 5) -> (5, window) then flatten to (450,)
            if should_invert:
                feat[:450] = (data_view.astype(np.float32) * -1.0).T.flatten()
            else:
                feat[:450] = data_view.astype(np.float32).T.flatten()

            # feat[450:] is already 0.0 due to np.zeros
            return feat
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _populate_entry_trend_backtest(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simplified and optimized logic for backtesting.
        """
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        lookback = 1000
        if self.config.get('deep_inference', False):
            df_input = dataframe
        else:
            df_input = dataframe.iloc[-2000:] if len(dataframe) > 2000 else dataframe

        def get_full_batch_q_values(df, side, model_num):
            window = 90
            cols = ['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']
            data = df[cols].values.astype(np.float32)
            if side == "SHORT" and ((model_num == 1 and self.short_1_is_mirror) or (model_num == 2 and self.short_2_is_mirror)):
                data = data * -1.0
            try:
                windows = sliding_window_view(data, window_shape=(window, 5)).squeeze(1)
                batch_windows = windows.transpose(0, 2, 1).reshape(len(windows), -1)
                add_feats = np.zeros((len(batch_windows), 4), dtype=np.float32)
                full_input = np.concatenate([batch_windows, add_feats], axis=1)
                t = torch.from_numpy(full_input).to(self.device)
                agent = (self.long_1_agent if model_num == 1 else self.long_2_agent) if side == "LONG" else (self.short_1_agent if model_num == 1 else self.short_2_agent)
                if agent is None: return None
                with torch.no_grad(): return agent.policy_net(t).cpu().numpy()
            except Exception as e:
                logger.error(f"Backtest batch inference failed: {e}")
                return None

        q_vals = {}
        if self.enable_long_1: q_vals['long_1'] = get_full_batch_q_values(df_input, "LONG", 1)
        if self.enable_long_2: q_vals['long_2'] = get_full_batch_q_values(df_input, "LONG", 2)
        if self.enable_short_1: q_vals['short_1'] = get_full_batch_q_values(df_input, "SHORT", 1)
        if self.enable_short_2: q_vals['short_2'] = get_full_batch_q_values(df_input, "SHORT", 2)

        offset = 89
        target_idx = df_input.index[offset:]
        n_results = len(target_idx)
        enter_long_vals = np.zeros(n_results, dtype=np.int8)
        enter_short_vals = np.zeros(n_results, dtype=np.int8)

        for i in range(n_results):
            candle_q = {name: vals[i:i+1] for name, vals in q_vals.items() if vals is not None}
            if not candle_q: continue
            decision = self._compute_ensemble_decision(candle_q, 0, has_long=False, has_short=False)
            enter_long_vals[i] = decision['enter_long']
            enter_short_vals[i] = decision['enter_short']

        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe.loc[target_idx, 'enter_long'] = enter_long_vals
        dataframe.loc[target_idx, 'enter_short'] = enter_short_vals
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        OPTIMIZED ENSEMBLE ENTRY LOGIC с параллельным инференсом
        """
        # Update dynamic epsilon once per candle based on current equity drawdown
        self._update_dynamic_epsilon()

        # --- PERIODIC SLOT UPDATE & LOGGING ---
        # Ensure slots are logged regularly for the dashboard, even if no trades occur
        if self.dynamic_slots_enabled:
            try:
                now = datetime.now(timezone.utc)
                if self.last_slot_update is None:
                    # Force immediate update on first run
                    self.last_slot_update = now - timedelta(days=1)
                
                if (now - self.last_slot_update).total_seconds() > self.slot_update_interval:
                    self._update_slot_allocation(now)
                    self.last_slot_update = now
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
        deep_inference = self.config.get('deep_inference', False)
        if self.config.get('runmode') in ['live', 'dry_run']:
            # Оптимизация: берем ровно 90 свечей для 1 предсказания (вместо 91 для 2)
            df_input = dataframe.iloc[-90:].copy()
        elif not deep_inference:
            lookback = 1000
            if len(dataframe) > lookback:
                df_input = dataframe.iloc[-lookback:].copy()
            else:
                df_input = dataframe
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
        else:
            df_input = dataframe
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
        
        # 3. ОПТИМИЗИРОВАННЫЙ инференс с кэшированием
        asset_name = metadata['pair'].split(':')[0].replace('/', '')
        
        # --- Q-VALUE CACHING FOR HYPEROPT ---
        # Ключ кэша: пара + время последней свечи.
        # Это гарантирует, что мы не пересчитываем нейросеть, если данные не изменились.
        last_date = dataframe.iloc[-1]['date']
        q_cache_key = (metadata['pair'], str(last_date))
        
        q_values = None
        with self.cache_lock:
            if q_cache_key in self.q_value_cache:
                q_values = self.q_value_cache[q_cache_key]
        
        if q_values is None:
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
            
            q_values = self._parallel_inference(inference_tasks)
            
            # Сохраняем в кэш
            with self.cache_lock:
                self.q_value_cache[q_cache_key] = q_values
        
        # Определяем размер батча из первого доступного результата
        batch_size = next(iter(q_values.values())).shape[0]
        
        # 5. Получение действий с порогом уверенности (Q-Threshold)
        # Фильтруем слабые сигналы, где Q(Action) почти равно Q(Hold)

        def get_action_with_threshold(name: str, target_action: Optional[int] = None):
            if q_values is None or name not in q_values:
                return np.zeros(batch_size, dtype=int), np.zeros(batch_size), 0.0
            
            # Берем порог из нормализации, так как глобального больше нет
            q_min = self.q_normalization.get(name, {}).get('q_min', 0.0)
            
            q = q_values[name]  # type: ignore
            
            if target_action is not None:
                # Forced check for specific action (Strict Mode)
                advantage = q[:, target_action] - q[:, 0]
                final_actions = np.where(advantage > q_min, target_action, 0)
            else:
                # Argmax check (Legacy/Flexible Mode)
                actions = np.argmax(q, axis=1)
                # Advantage = Q(Selected) - Q(Hold)
                advantage = q[np.arange(len(q)), actions] - q[:, 0]
                final_actions = np.where(advantage > q_min, actions, 0)
                
            return final_actions, advantage, q_min

        action_long_1, adv_long_1, th_l1 = get_action_with_threshold("long_1", target_action=1)
        action_long_2, adv_long_2, th_l2 = get_action_with_threshold("long_2", target_action=1)
        
        # Mirror mode models (and all models in config trained as SHORT) 
        # use index 1 for their primary 'ENTRY' action
        action_short_1, adv_short_1, th_s1 = get_action_with_threshold("short_1", target_action=1)
        action_short_2, adv_short_2, th_s2 = get_action_with_threshold("short_2", target_action=1)

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

        for i in range(n_predictions):
            # Собираем действия для текущей свечи (Raw actions: 0 or 1)
            # НОВЫЙ МЕТОД: Нормируешь → суммируешь → сравниваешь разницу → фильтруешь по ε
            decision = self._compute_ensemble_decision(
                q_values, i, has_long, has_short
            )

            # --- Volume Filter ---
            # Применяется после генерации сигнала, чтобы можно было залогировать причину
            if volume_vals is not None:
                min_volume = self.min_quote_volume_usd.value
                current_volume = volume_vals[i]  # type: ignore
                if current_volume < min_volume:
                    if decision['enter_long'] == 1 or decision['enter_short'] == 1:
                        decision['reason'] += f" | ⛔ Filtered by Volume ({current_volume:.0f} < {min_volume:.0f})"
                    decision['enter_long'] = 0
                    decision['enter_short'] = 0

            # --- Regime Filter (Post-Ensemble) ---
            if is_backtest:
                # Backtest: check regimes per candle
                if decision['enter_long'] == 1:
                    if self.use_global_regime_filter and (regime_global_vals is None or regime_global_vals[i] <= 0):
                        decision['reason'] += f" | ⛔ Filtered by ST (Global Bear/Flat)"
                        decision['enter_long'] = 0
                    if self.use_local_regime_filter and (regime_local_vals is None or regime_local_vals[i] <= 0) and decision['enter_long'] == 1:
                        decision['reason'] += f" | ⛔ Filtered by ST (Local Bear/Flat)"
                        decision['enter_long'] = 0
                
                if decision['enter_short'] == 1:
                    if self.use_global_regime_filter and (regime_global_vals is None or regime_global_vals[i] >= 0):
                        decision['reason'] += f" | ⛔ Filtered by ST (Global Bull/Flat)"
                        decision['enter_short'] = 0
                    if self.use_local_regime_filter and (regime_local_vals is None or regime_local_vals[i] >= 0) and decision['enter_short'] == 1:
                        decision['reason'] += f" | ⛔ Filtered by ST (Local Bull/Flat)"
                        decision['enter_short'] = 0
            else:
                # Live/Dry-run: use pre-calculated flags as a final check
                if not allow_long and decision['enter_long'] == 1:
                    reason_str = "Regime"
                    if self.use_global_regime_filter and 'st_regime_global' in dataframe.columns and dataframe['st_regime_global'].iloc[-1] <= 0:
                        reason_str = "Global Bear/Flat"
                    elif self.use_local_regime_filter and 'st_regime_local' in dataframe.columns and dataframe['st_regime_local'].iloc[-1] <= 0:
                        reason_str = "Local Bear/Flat"
                    decision['reason'] += f" | ⛔ Filtered by ST ({reason_str})"
                    decision['enter_long'] = 0

                if not allow_short and decision['enter_short'] == 1:
                    decision['reason'] += f" | ⛔ Filtered by ST (Regime)"
                    decision['enter_short'] = 0

            # Сбор статистики
            if decision['enter_long'] or decision['enter_short']:
                self.conflict_stats['total_signals'] += 1
                if 'Conflict' in decision['reason']:
                    self.conflict_stats['conflicts'] += 1
                elif decision['enter_long']:
                    self.conflict_stats['long_entries'] += 1
                elif decision['enter_short']:
                    self.conflict_stats['short_entries'] += 1
            
            # Логирование
            # Логируем только последние 2 свечи (0 и 1)
            if i >= n_predictions - 2:
                if decision['enter_long'] or decision['enter_short']:
                    self.logger.info(f"[SIGNAL] {metadata['pair']} ENTRY SIGNAL: {decision['reason']}")
            
            # Записываем в массив (быстро)
            enter_long_vals[i] = decision['enter_long']
            enter_short_vals[i] = decision['enter_short']
        
        # Mass assignment (один раз для всех строк)
        dataframe.loc[target_idx, 'enter_long'] = enter_long_vals
        dataframe.loc[target_idx, 'enter_short'] = enter_short_vals

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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