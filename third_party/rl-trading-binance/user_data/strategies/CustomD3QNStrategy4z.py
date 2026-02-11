import sys
import logging
import logging.handlers
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Optional, List, Any
from collections import deque


try:
    from freqtrade.persistence import Trade  # type: ignore
except ImportError:
    class Trade: pass

from datetime import datetime

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
    from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter  # type: ignore
except ImportError:
    logging.getLogger(__name__).error("Could not import freqtrade.strategy")
    class IStrategy:
        def __init__(self, config: dict, **kwargs):
            self.config = config
    class DecimalParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0.0)
    class IntParameter:
        def __init__(self, *args, **kwargs): self.value = kwargs.get('default', 0)

logger = logging.getLogger(__name__)

# Agent imports
try:
    from agent import D3QN_PER_Agent
except ImportError as e:
    logger.error(f"CRITICAL: Could not import D3QN Agent! Check path: {project_root}")
    raise e


class CustomD3QNStrategy4z(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short: bool = True  # Это критично для Futures режима
    startup_candle_count: int = 200
    
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
    d0 = DecimalParameter(0.01, 0.10, default=0.075, space='stoploss', load=True)
    d_min = DecimalParameter(0.001, 0.05, default=0.0008, space='stoploss', load=True)
    hysteresis = DecimalParameter(0.00005, 0.01, default=0.00075, space='stoploss', load=True)
    
    # Hyperoptable Voting Thresholds
    rl_long_threshold_opt = IntParameter(1, 2, default=1, space='buy', optimize=True, load=True)
    rl_short_threshold_opt = IntParameter(1, 2, default=1, space='sell', optimize=True, load=True)

    plot_config = {
        'main_plot': {},
        'subplots': {}
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Принудительно включаем шорты
        self.can_short = True
        
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
                logger.info(f"🔄 Log rotation enabled for {h.baseFilename} (Daily at midnight)")
                
        except Exception as e:
            logger.warning(f"⚠️ Log rotation setup failed: {e}")

        # === CPU ОПТИМИЗАЦИИ ===
        # 1. Установить количество потоков для PyTorch
        num_cpu_threads = config.get('cpu_threads', 4)  # по умолчанию 4 потока
        try:
            torch.set_num_threads(num_cpu_threads)
            torch.set_num_interop_threads(num_cpu_threads)
        except RuntimeError as e:
            logger.warning(f"⚠️ Could not set torch threads (already initialized?): {e}")
        
        self.device = torch.device("cpu")
        
        self.logger = logging.getLogger(__name__)
        # 2. ThreadPool для параллельного inference
        self.executor = ThreadPoolExecutor(max_workers=num_cpu_threads)
        
        # 3. Кэш для feature tensors (экономим на preprocessing)
        self.feature_cache = {}
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

        # === DYNAMIC SLOT ALLOCATION ===
        self.dynamic_slots_cfg = config.get('dynamic_slots', {})
        self.dynamic_slots_enabled = self.dynamic_slots_cfg.get('enabled', False)
        self.total_slots = config.get('max_open_trades', 100)  # Используем глобальный параметр
        self.min_slots_per_side = self.dynamic_slots_cfg.get('min_slots_per_side', 10)
        self.aggression_factor = self.dynamic_slots_cfg.get('aggression_factor', 1.5)
        self.slot_update_interval = self.dynamic_slots_cfg.get('update_interval_sec', 300)

        self.max_long_slots = self.total_slots // 2 if self.total_slots > 0 else 50
        self.max_short_slots = self.total_slots - self.max_long_slots if self.total_slots > 0 else 50
        self.last_slot_update: Optional[datetime] = None
        self.slot_history = deque(maxlen=100)

        logger.info(f"🎰 Dynamic Slots: {'ENABLED' if self.dynamic_slots_enabled else 'DISABLED'} | Total: {self.total_slots}")

        # --- ПУТИ К 4 МОДЕЛЯМ ---
        # Long Model 1:
        self.long_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260125_time_033653"
        self.long_1_model_pth = self.long_1_model_dir / "best.pth"
        
        # Long Model 2:
        self.long_2_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260201_time_131607_no tsl"
        self.long_2_model_pth = self.long_2_model_dir / "best.pth"
        
        # Short Model 1:
        self.short_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260126_time_214322"
        self.short_1_model_pth = self.short_1_model_dir / "best.pth"
        
        # Short Model 2:
        self.short_2_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_z_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260207_time_144255"
        self.short_2_model_pth = self.short_2_model_dir / "best.pth"
        
        # --- ВКЛЮЧЕНИЕ/ОТКЛЮЧЕНИЕ МОДЕЛЕЙ ---
        self.enable_long_1 = config.get('rl_enable_long_1', True)
        self.enable_long_2 = config.get('rl_enable_long_2', True)
        self.enable_short_1 = config.get('rl_enable_short_1', True)
        self.enable_short_2 = config.get('rl_enable_short_2', True)
        
        # --- ЗАГРУЗКА КОНФИГОВ ---
        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING 2+2 ENSEMBLE SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"🔌 Active Models: L1={self.enable_long_1}, L2={self.enable_long_2}, S1={self.enable_short_1}, S2={self.enable_short_2}")
        
        # Long 1
        if self.enable_long_1:
            cfg_file_long_1 = self._find_config_file(self.long_1_model_dir)
            if not cfg_file_long_1:
                raise FileNotFoundError(f"Config not found in {self.long_1_model_dir}")
            logger.info(f"✓ Loading LONG_1 config from {cfg_file_long_1}")
            self.cfg_long_1 = self._load_py_config(cfg_file_long_1)
        else:
            self.cfg_long_1 = None
        
        # Long 2
        if self.enable_long_2:
            cfg_file_long_2 = self._find_config_file(self.long_2_model_dir)
            if not cfg_file_long_2:
                raise FileNotFoundError(f"Config not found in {self.long_2_model_dir}")
            logger.info(f"✓ Loading LONG_2 config from {cfg_file_long_2}")
            self.cfg_long_2 = self._load_py_config(cfg_file_long_2)
        else:
            self.cfg_long_2 = None
        
        # Short 1
        if self.enable_short_1:
            cfg_file_short_1 = self._find_config_file(self.short_1_model_dir)
            if not cfg_file_short_1:
                raise FileNotFoundError(f"Config not found in {self.short_1_model_dir}")
            logger.info(f"✓ Loading SHORT_1 config from {cfg_file_short_1}")
            self.cfg_short_1 = self._load_py_config(cfg_file_short_1)
        else:
            self.cfg_short_1 = None
        
        # Short 2
        if self.enable_short_2:
            cfg_file_short_2 = self._find_config_file(self.short_2_model_dir)
            if not cfg_file_short_2:
                raise FileNotFoundError(f"Config not found in {self.short_2_model_dir}")
            logger.info(f"✓ Loading SHORT_2 config from {cfg_file_short_2}")
            self.cfg_short_2 = self._load_py_config(cfg_file_short_2)
        else:
            self.cfg_short_2 = None
        
        # --- ОПРЕДЕЛЕНИЕ РЕЖИМА MIRROR MODE ---
        # Определяем из конфига модели
        self.short_1_is_mirror = getattr(self.cfg_short_1.market, 'mirror_mode', False) if self.cfg_short_1 else False
        self.short_2_is_mirror = getattr(self.cfg_short_2.market, 'mirror_mode', False) if self.cfg_short_2 else False
        
        logger.info(f"ℹ️ SHORT_1 Mirror Mode: {self.short_1_is_mirror} (From Config)")
        logger.info(f"ℹ️ SHORT_2 Mirror Mode: {self.short_2_is_mirror} (From Config)")
        
        # --- НАСТРОЙКИ АНСАМБЛЯ V2 ---
        self.ensemble_cfg = config.get('rl_ensemble', {})
        # Base epsilon from config (used as baseline for dynamic epsilon)
        self.epsilon_threshold = self.ensemble_cfg.get('epsilon_threshold', 0.15)
        # Effective epsilon actually used for thresholding (will be updated dynamically)
        self.epsilon_threshold_eff: float = self.epsilon_threshold
        # Раздельные эффективные eps для лонгов и шортов
        self.epsilon_threshold_eff_long: float = self.epsilon_threshold
        self.epsilon_threshold_eff_short: float = self.epsilon_threshold
        # Максимальная наблюдаемая equity по unrealized PnL для лонгов и шортов
        self.equity_max_long: float = 0.0
        self.equity_max_short: float = 0.0
        # Старое поле equity_max оставляем для обратной совместимости (не используется напрямую)
        self.equity_max: float = 0.0

        self.enable_veto = config.get('rl_enable_veto', False)
        self.rl_long_threshold = config.get('rl_long_threshold', 1)
        self.rl_short_threshold = config.get('rl_short_threshold', 1)
        self.q_normalization = self.ensemble_cfg.get('q_normalization', {})
        self.config_update_interval = self.ensemble_cfg.get('q_update_interval', config.get('q_update_interval', 14400))

        logger.info(f"🗳️ Ensemble Config: Epsilon={self.epsilon_threshold} | UpdateInterval={self.config_update_interval}s")

        # --- ИНИЦИАЛИЗАЦИЯ 4 АГЕНТОВ ---
        logger.info("📦 Creating agents...")
        
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
        logger.info("🔧 Optimizing models for CPU inference...")
        
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
        
        logger.info("✅ CPU optimizations applied")
        
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
        from pydantic import ValidationError

        spec = importlib.util.spec_from_file_location("mod_cfg", file_path)
        mod = importlib.util.module_from_spec(spec)
        
        # Хак для обратной совместимости: 
        # Если в загружаемом файле есть обращение к несуществующим полям Pydantic,
        # нам нужно это перехватить. 
        try:
            spec.loader.exec_module(mod)
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
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Окно нормализации из обучения
        window = 90
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            rolling = dataframe[col].rolling(window=window, min_periods=window)
            mean = rolling.mean()
            std = rolling.std(ddof=0)
            # Z-score: (x - mean) / std
            dataframe[f'{col}_z'] = (dataframe[col] - mean) / (std + 1e-8)

        # Заполняем NaN нулями (начало датафрейма), чтобы модель не получала inf/nan
        z_cols = [f'{col}_z' for col in ohlcv_cols]
        dataframe[z_cols] = dataframe[z_cols].fillna(0.0)

        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs):
        if trade.open_date_utc:
            duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
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

        if calc_p <= FEE_BUF:
            d_eff = d0_val
        else:
            d_eff = d0_val - (calc_p - FEE_BUF)
            d_eff = max(d_min_val, d_eff)
        
        return -d_eff

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str) -> Optional[torch.Tensor]:
        # 1. Проверка длины
        if len(dataframe) < 90:
            return None
            
        # 2. Выбор Z-колонок
        cols = ['open_z', 'high_z', 'low_z', 'close_z', 'volume_z']
        
        # 3. Sliding window
        # (N, 5)
        z_data = dataframe[cols].values.astype(np.float32)
        
        # (N, 5) -> (Batch, 90, 5)
        windows = sliding_window_view(z_data, window_shape=90, axis=0)
        
        # 4. Inversion logic (Mirror Mode)
        should_invert = False
        if side == "SHORT":
            if (model_num == 1 and self.short_1_is_mirror) or (model_num == 2 and self.short_2_is_mirror):
                should_invert = True
        
        if should_invert:
            windows = windows * -1.0

        # 5. Flatten & Extra Features
        batch_size = windows.shape[0]
        flat_feats = windows.reshape(batch_size, -1)
        
        add_feats = np.zeros((batch_size, 4), dtype=np.float32)
        # Set time_remaining (index 3) to 1.0 (start of session)
        add_feats[:, 3] = 1.0
        combined = np.concatenate([flat_feats, add_feats], axis=1)
        
        # Return tensor on device
        return torch.as_tensor(combined, device=self.device, dtype=torch.float32)

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
        def single_inference(tensor, agent):
            try:
                with torch.no_grad():
                    return agent.policy_net(tensor).cpu().numpy()
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                return np.zeros((tensor.shape[0], agent.action_dim))
        
        futures = []
        for tensor, agent, name in tensors_and_agents:
            future = self.executor.submit(single_inference, tensor, agent)
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
                logger.warning(f"⚠️ Config update skipped: {config_path} not found")
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            updated = False
            norm_cfg = data.get('rl_ensemble', {}).get('q_normalization', {})
            
            for name, history in self.adv_history.items():
                if len(history) < 100: 
                    logger.info(f"⏳ {name}: Insufficient history for Q-update ({len(history)}/100)")
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
                logger.info(f"⚖️ Auto-tuned {name}: q_min={new_min}, q_max={new_max}")
            
            if updated:
                if 'rl_ensemble' not in data: data['rl_ensemble'] = {}
                data['rl_ensemble']['q_normalization'] = norm_cfg
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                self.q_normalization = norm_cfg
                logger.info(f"💾 Config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to auto-tune config: {e}")

    def _get_pnl_from_freqtrade(self) -> tuple:
        """
        Получает ТОЛЬКО Unrealized PnL (открытые позиции)
        для мгновенной реакции на изменение рынка
        Возвращает (pnl_long_usdt, pnl_short_usdt)
        """
        try:
            # === ОТКРЫТЫЕ ПОЗИЦИИ (Unrealized PnL) ===
            open_trades = Trade.get_open_trades()

            pnl_long_open = 0.0
            pnl_short_open = 0.0

            for t in open_trades:
                try:
                    if t.close_rate_requested:
                        profit_usdt = t.calc_profit(rate=t.close_rate_requested)
                    else:
                        if hasattr(self, 'dp') and self.dp:
                            try:
                                dataframe, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
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
                    pnl_short_open += profit_usdt
                else:
                    pnl_long_open += profit_usdt

            # Возвращаем ТОЛЬКО Unrealized PnL
            logger.debug(
                f"PnL (Unrealized Only): "
                f"Long Open={pnl_long_open:.2f} | "
                f"Short Open={pnl_short_open:.2f}"
            )

            return pnl_long_open, pnl_short_open

        except Exception as e:
            logger.error(f"Failed to calculate PnL: {e}")
            return 0.0, 0.0

    def _update_dynamic_epsilon(self) -> None:
        """
        Update effective epsilon based on current equity drawdown.

        Equity is approximated as the sum of unrealized PnL from the long and short books.
        As drawdown from the maximum observed equity increases, the effective epsilon
        increases smoothly, requiring stronger model advantages to cast a real vote.
        When drawdown decreases back towards zero, epsilon gradually returns towards
        the base value from config.
        """
        # Skip dynamic epsilon update in backtesting/hyperopt to avoid DB calls
        if self.config.get('runmode') not in ['live', 'dry_run']:
            return

        try:
            pnl_long, pnl_short = self._get_pnl_from_freqtrade()
            equity_long = pnl_long
            equity_short = pnl_short

            # Reset dynamic epsilon if there are no open trades
            try:
                from freqtrade.persistence import Trade  # type: ignore
                open_trades_q = Trade.get_open_trades()
                if hasattr(open_trades_q, "all"):
                    open_trades = open_trades_q.all()
                else:
                    open_trades = open_trades_q
            except Exception:
                open_trades = []

            if not open_trades:
                # Вне рынка: сбрасываем состояние drawdown и возвращаемся к базовому epsilon
                self.equity_max = 0.0
                self.equity_max_long = 0.0
                self.equity_max_short = 0.0
                self.epsilon_threshold_eff = self.epsilon_threshold
                self.epsilon_threshold_eff_long = self.epsilon_threshold
                self.epsilon_threshold_eff_short = self.epsilon_threshold
                if self.config.get('runmode') in ['live', 'dry_run']:
                    self.logger.debug(f"EPS-DD | reset (no open trades) | base={self.epsilon_threshold:.3f} | effL={self.epsilon_threshold_eff_long:.3f} effS={self.epsilon_threshold_eff_short:.3f}")
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
            if self.equity_max_long > 0.0:
                dd_long = (self.equity_max_long - equity_long) / self.equity_max_long
            else:
                dd_long = 0.0
            if self.equity_max_short > 0.0:
                dd_short = (self.equity_max_short - equity_short) / self.equity_max_short
            else:
                dd_short = 0.0

            dd_long = max(0.0, min(dd_long, 1.0))
            dd_short = max(0.0, min(dd_short, 1.0))

            # Linear sensitivity: epsilon_target = epsilon_0 * (1 + k * DD) — раздельно для long/short
            k = 6.0
            epsilon_target_long = self.epsilon_threshold * (1.0 + k * dd_long)
            epsilon_target_short = self.epsilon_threshold * (1.0 + k * dd_short)

            # Clamp epsilon_target into [0.1, 1.0] — как было, раздельно для long/short
            epsilon_target_long = float(min(max(epsilon_target_long, 0.1), 1.0))
            epsilon_target_short = float(min(max(epsilon_target_short, 0.1), 1.0))

            # Smooth update via EMA to avoid abrupt jumps — раздельно для long/short
            alpha = 0.4

            if not hasattr(self, "epsilon_threshold_eff_long") or self.epsilon_threshold_eff_long <= 0.0:
                self.epsilon_threshold_eff_long = self.epsilon_threshold
            if not hasattr(self, "epsilon_threshold_eff_short") or self.epsilon_threshold_eff_short <= 0.0:
                self.epsilon_threshold_eff_short = self.epsilon_threshold

            self.epsilon_threshold_eff_long = (
                (1.0 - alpha) * self.epsilon_threshold_eff_long + alpha * epsilon_target_long
            )
            self.epsilon_threshold_eff_short = (
                (1.0 - alpha) * self.epsilon_threshold_eff_short + alpha * epsilon_target_short
            )

            # Глобальный epsilon_threshold_eff оставляем как среднее (для обратной совместимости, если где-то используется)
            self.epsilon_threshold_eff = 0.5 * (self.epsilon_threshold_eff_long + self.epsilon_threshold_eff_short)

            if self.config.get('runmode') in ['live', 'dry_run']:
                self.logger.debug(
                    f"EPS-DD | ddL={dd_long:.3f} ddS={dd_short:.3f} | base={self.epsilon_threshold:.3f} | "
                    f"effL={self.epsilon_threshold_eff_long:.3f} effS={self.epsilon_threshold_eff_short:.3f}"
                )
        except Exception as e:
            # Fail-open: fall back to base epsilon
            self.logger.warning(f"Dynamic epsilon update failed: {e}. Falling back to base epsilon.")
            self.epsilon_threshold_eff = self.epsilon_threshold
            self.epsilon_threshold_eff_long = self.epsilon_threshold
            self.epsilon_threshold_eff_short = self.epsilon_threshold

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

            penalty_ratio = (loss_short / total) ** self.aggression_factor
            long_ratio = 0.8 + 0.15 * penalty_ratio
            reason = f"Long profitable (+{pnl_long:.0f}), Short losing (-{loss_short:.0f})"

        elif pnl_short > 0 and pnl_long < 0:
            # Зеркально
            profit_short = pnl_short
            loss_long = abs(pnl_long)
            total = profit_short + loss_long

            penalty_ratio = (loss_long / total) ** self.aggression_factor
            long_ratio = 0.2 - 0.15 * penalty_ratio
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
                long_ratio = loss_short / total_loss
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
        logger.info(f"🎰 SLOTS: L={self.max_long_slots} ({pnl_long:+.1f} USDT) | S={self.max_short_slots} ({pnl_short:+.1f} USDT) | {reason}")

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
            logger.warning(f"⚠️ Degenerate Q stats for {model_name}: q_min={q_min}, q_max={q_max}. Model is likely a zombie.")
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
        veto_long_count = 0
        veto_short_count = 0
        details = []

        # --- 1. Подсчет голосов LONG ---
        def check_vote(name, action_idx):
            if name not in q_values: return 0, False, 0.0
            q_hold = q_values[name][idx, 0]
            q_action = q_values[name][idx, action_idx]
            adv = q_action - q_hold

            cfg = self.q_normalization.get(name, {})
            q_min = cfg.get('q_min', 0.0)
            q_max = cfg.get('q_max', None)

            if q_max is None or q_max <= q_min:
                logger.warning(f"⚠️ Degenerate Q stats for {name}, excluding zombie model.")
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
            v, active, norm = check_vote("long_1", 1)
            votes_long += v
            if v: veto_long_count += 1
            if v: details.append(f"L1({norm:.2f})")

        if self.enable_long_2:
            v, active, norm = check_vote("long_2", 1)
            votes_long += v
            if v: veto_long_count += 1
            if v: details.append(f"L2({norm:.2f})")

        # --- 2. Подсчет голосов SHORT ---
        if self.enable_short_1:
            action_idx = 1 if self.short_1_is_mirror else 2
            v, active, norm = check_vote("short_1", action_idx)
            votes_short += v
            if v: veto_short_count += 1
            if v: details.append(f"S1({norm:.2f})")

        if self.enable_short_2:
            action_idx = 1 if self.short_2_is_mirror else 2
            v, active, norm = check_vote("short_2", action_idx)
            votes_short += v
            if v: veto_short_count += 1
            if v: details.append(f"S2({norm:.2f})")

        result = {
            'enter_long': 0,
            'enter_short': 0,
            'reason': f"Votes L:{votes_long}/{thresh_long} S:{votes_short}/{thresh_short} [{' '.join(details)}]"
        }

        if has_long or has_short:
            result['reason'] += " | Position exists"
            return result

        # --- 3. Принятие решения ---
        long_signal = votes_long >= thresh_long
        short_signal = votes_short >= thresh_short

        if long_signal and short_signal:
            result['reason'] += " | CONFLICT (Both signals)"
            return result

        if long_signal:
            if self.enable_veto and veto_short_count > 0:
                result['reason'] += " | LONG Vetoed (Short vote present)"
            else:
                result['enter_long'] = 1
                result['reason'] += " | LONG Signal"

        elif short_signal and self.can_short:
            if self.enable_veto and veto_long_count > 0:
                result['reason'] += " | SHORT Vetoed (Long vote present)"
            else:
                result['enter_short'] = 1
                result['reason'] += " | SHORT Signal"

        return result

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        # Разрешаем проверку слотов в бэктесте для полнофункциональной симуляции
        if self.config.get('runmode') not in ['live', 'dry_run']:
            return True
        
        try:
            from freqtrade.persistence import Trade  # type: ignore
            from datetime import timezone
            
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
                    if self.last_slot_update is None or \
                       (current_time - self.last_slot_update).total_seconds() > self.slot_update_interval:
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
                    self.logger.info(f"🚫 LONG LIMIT: {pair} {current_longs}/{self.max_long_slots}")
                    return False
            elif side == "short":
                if current_shorts >= self.max_short_slots:
                    self.logger.info(f"🚫 SHORT LIMIT: {pair} {current_shorts}/{self.max_short_slots}")
                    return False
        except Exception as e:
            self.logger.error(f"Error in confirm_trade_entry: {e}")
            return True
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, sell_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        if trade.id in self.tsl_memory:
            del self.tsl_memory[trade.id]
        return True
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        OPTIMIZED ENSEMBLE ENTRY LOGIC с параллельным инференсом
        """
        # Update dynamic epsilon once per candle based on current equity drawdown
        self._update_dynamic_epsilon()

        # 1. Базовая защита
        if len(dataframe) < self.startup_candle_count:
            return dataframe
        
        # 2. Оптимизация инференса
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
        
        tensor_long_1 = self.get_model_input_cached(df_input, metadata['pair'], side="LONG", model_num=1, asset_name=asset_name) if self.enable_long_1 else None
        tensor_long_2 = self.get_model_input_cached(df_input, metadata['pair'], side="LONG", model_num=2, asset_name=asset_name) if self.enable_long_2 else None
        tensor_short_1 = self.get_model_input_cached(df_input, metadata['pair'], side="SHORT", model_num=1, asset_name=asset_name) if self.enable_short_1 else None
        tensor_short_2 = self.get_model_input_cached(df_input, metadata['pair'], side="SHORT", model_num=2, asset_name=asset_name) if self.enable_short_2 else None
        
        # Проверяем, что все ВКЛЮЧЕННЫЕ модели получили данные
        missing_data = False
        if self.enable_long_1 and tensor_long_1 is None: missing_data = True
        if self.enable_long_2 and tensor_long_2 is None: missing_data = True
        if self.enable_short_1 and tensor_short_1 is None: missing_data = True
        if self.enable_short_2 and tensor_short_2 is None: missing_data = True

        if missing_data:
            return dataframe
        
        # Sanity Check: Ensure Short tensor is not identical to Long tensor (should be inverted)
        if tensor_long_1 is not None and tensor_short_1 is not None:
            # Предупреждаем только если включен mirror_mode. В обычном режиме они И ДОЛЖНЫ быть одинаковыми.
            if self.short_1_is_mirror and torch.equal(tensor_long_1, tensor_short_1):
                logger.warning(f"🚨 CRITICAL: LONG_1 and SHORT_1 tensors are IDENTICAL for {metadata['pair']}! Inversion failed?")

        # 4. ПАРАЛЛЕЛЬНЫЙ INFERENCE для всех 4 моделей одновременно
        inference_tasks = []
        if self.enable_long_1 and tensor_long_1 is not None:
            inference_tasks.append((tensor_long_1, self.long_1_agent, "long_1"))
        if self.enable_long_2 and tensor_long_2 is not None:
            inference_tasks.append((tensor_long_2, self.long_2_agent, "long_2"))
        if self.enable_short_1 and tensor_short_1 is not None:
            inference_tasks.append((tensor_short_1, self.short_1_agent, "short_1"))
        if self.enable_short_2 and tensor_short_2 is not None:
            inference_tasks.append((tensor_short_2, self.short_2_agent, "short_2"))
            
        if not inference_tasks:
            return dataframe
        
        q_values = self._parallel_inference(inference_tasks)
        
        # Определяем размер батча из первого доступного результата
        batch_size = next(iter(q_values.values())).shape[0]
        
        # 5. Получение действий с порогом уверенности (Q-Threshold)
        # Фильтруем слабые сигналы, где Q(Action) почти равно Q(Hold)

        def get_action_with_threshold(name, target_action=None):
            if name not in q_values:
                return np.zeros(batch_size, dtype=int), np.zeros(batch_size), 0.0
            
            # Берем порог из нормализации, так как глобального больше нет
            q_min = self.q_normalization.get(name, {}).get('q_min', 0.0)
            
            q = q_values[name]
            
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
        
        s1_act = 1 if self.short_1_is_mirror else 2
        action_short_1, adv_short_1, th_s1 = get_action_with_threshold("short_1", target_action=s1_act)
        
        s2_act = 1 if self.short_2_is_mirror else 2
        action_short_2, adv_short_2, th_s2 = get_action_with_threshold("short_2", target_action=s2_act)

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
                vote_mark = "🟢 VOTE" if vote else "NO"

                logger.info(f"{metadata['pair']} L1: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

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
                vote_mark = "🟢 VOTE" if vote else "NO"

                logger.info(f"{metadata['pair']} L2: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

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
                vote_mark = "🟢 VOTE" if vote else "NO"

                logger.info(f"{metadata['pair']} S1: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

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
                vote_mark = "🟢 VOTE" if vote else "NO"

                logger.info(f"{metadata['pair']} S2: adv={adv:.5f} vs thr={thr:.5f} | norm={norm:.2f} | {vote_mark} | act={a} ({a_str})")

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
            open_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True)]).first()
            has_long = open_trade.is_short is False if open_trade else False
            has_short = open_trade.is_short is True if open_trade else False
        except Exception:
            pass

        for i in range(n_predictions):
            # Собираем действия для текущей свечи (Raw actions: 0 or 1)
            # НОВЫЙ МЕТОД: Нормируешь → суммируешь → сравниваешь разницу → фильтруешь по ε
            decision = self._compute_ensemble_decision(
                q_values, i, has_long, has_short
            )
            
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
                    logger.info(f"📊 {metadata['pair']} ENTRY SIGNAL: {decision['reason']}")
            
            # Записываем в DF
            dataframe.loc[target_idx[i], 'enter_long'] = decision['enter_long']
            dataframe.loc[target_idx[i], 'enter_short'] = decision['enter_short']
        
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
        logger.warning(f"Leverage not found for {pair} in config. Falling back to proposed: {proposed_leverage}")
        return proposed_leverage