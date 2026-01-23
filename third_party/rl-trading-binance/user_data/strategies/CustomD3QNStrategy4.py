import sys
import json
import logging
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ThreadPoolExecutor
import threading

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
    class IStrategy: pass
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


class CustomD3QNStrategy4(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short = True
    startup_candle_count: int = 100
    
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
    d_min = DecimalParameter(0.001, 0.05, default=0.01, space='stoploss', load=True)
    hysteresis = DecimalParameter(0.001, 0.02, default=0.002, space='stoploss', load=True)
    
    plot_config = {
        'main_plot': {},
        'subplots': {}
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # === CPU ОПТИМИЗАЦИИ ===
        # 1. Установить количество потоков для PyTorch
        num_cpu_threads = config.get('cpu_threads', 4)  # по умолчанию 4 потока
        torch.set_num_threads(num_cpu_threads)
        torch.set_num_interop_threads(num_cpu_threads)
        
        self.device = torch.device("cpu")
        
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
        
        # Статистика конфликтов
        self.conflict_stats = {
            'total_signals': 0,
            'conflicts': 0,
            'long_entries': 0,
            'short_entries': 0,
        }
        
        # --- ПУТИ К 4 МОДЕЛЯМ ---
        # Long Model 1: PPO trending
        self.long_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260118_time_220542"
        self.long_1_model_pth = self.long_1_model_dir / "best.pth"
        
        # Long Model 2: A2C mean-reversion (используем ту же модель для примера, замените на вашу вторую)
        self.long_2_model_dir = self.project_root / "output/alpha_seed_405_ohlcv_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260121_time_232557"
        self.long_2_model_pth = self.long_2_model_dir / "best.pth"
        
        # Short Model 1: SAC bearish trending
        self.short_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260118_time_225844"
        self.short_1_model_pth = self.short_1_model_dir / "best.pth"
        
        # Short Model 2: PPO short mean-reversion (используем ту же модель для примера, замените на вашу вторую)
        self.short_2_model_dir = self.project_root / "output/alpha_seed_405_ohlcv_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260121_time_223959"
        self.short_2_model_pth = self.short_2_model_dir / "best.pth"
        
        # --- ЗАГРУЗКА КОНФИГОВ ---
        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING 2+2 ENSEMBLE SYSTEM")
        logger.info("=" * 60)
        
        # Long 1
        cfg_file_long_1 = self._find_config_file(self.long_1_model_dir)
        if not cfg_file_long_1:
            raise FileNotFoundError(f"Config not found in {self.long_1_model_dir}")
        logger.info(f"✓ Loading LONG_1 config from {cfg_file_long_1}")
        self.cfg_long_1 = self._load_py_config(cfg_file_long_1)
        
        # Long 2
        cfg_file_long_2 = self._find_config_file(self.long_2_model_dir)
        if not cfg_file_long_2:
            raise FileNotFoundError(f"Config not found in {self.long_2_model_dir}")
        logger.info(f"✓ Loading LONG_2 config from {cfg_file_long_2}")
        self.cfg_long_2 = self._load_py_config(cfg_file_long_2)
        
        # Short 1
        cfg_file_short_1 = self._find_config_file(self.short_1_model_dir)
        if not cfg_file_short_1:
            raise FileNotFoundError(f"Config not found in {self.short_1_model_dir}")
        logger.info(f"✓ Loading SHORT_1 config from {cfg_file_short_1}")
        self.cfg_short_1 = self._load_py_config(cfg_file_short_1)
        
        # Short 2
        cfg_file_short_2 = self._find_config_file(self.short_2_model_dir)
        if not cfg_file_short_2:
            raise FileNotFoundError(f"Config not found in {self.short_2_model_dir}")
        logger.info(f"✓ Loading SHORT_2 config from {cfg_file_short_2}")
        self.cfg_short_2 = self._load_py_config(cfg_file_short_2)
        
        # --- ЗАГРУЗКА NORM_STATS ---
        self.norm_stats_long_1 = self._load_norm_stats(self.long_1_model_dir)
        self.norm_stats_long_2 = self._load_norm_stats(self.long_2_model_dir)
        self.norm_stats_short_1 = self._load_norm_stats(self.short_1_model_dir)
        self.norm_stats_short_2 = self._load_norm_stats(self.short_2_model_dir)
        
        # --- ИНИЦИАЛИЗАЦИЯ 4 АГЕНТОВ ---
        logger.info("📦 Creating agents...")
        self.long_1_agent = self._create_agent_from_config(self.cfg_long_1)
        self.long_2_agent = self._create_agent_from_config(self.cfg_long_2)
        self.short_1_agent = self._create_agent_from_config(self.cfg_short_1)
        self.short_2_agent = self._create_agent_from_config(self.cfg_short_2)
        
        # --- ЗАГРУЗКА ВЕСОВ ---
        self._load_weights(self.long_1_agent, self.long_1_model_pth, "LONG_1")
        self._load_weights(self.long_2_agent, self.long_2_model_pth, "LONG_2")
        self._load_weights(self.short_1_agent, self.short_1_model_pth, "SHORT_1")
        self._load_weights(self.short_2_agent, self.short_2_model_pth, "SHORT_2")
        
        # === ОПТИМИЗАЦИЯ МОДЕЛЕЙ ДЛЯ INFERENCE ===
        # После загрузки весов, оптимизируем модели
        logger.info("🔧 Optimizing models for CPU inference...")
        
        # Переводим в eval mode и оптимизируем
        for agent_name, agent in [
            ("LONG_1", self.long_1_agent),
            ("LONG_2", self.long_2_agent),
            ("SHORT_1", self.short_1_agent),
            ("SHORT_2", self.short_2_agent)
        ]:
            agent.policy_net.eval()
            
            # Отключаем grad для всех параметров (экономит память и время)
            for param in agent.policy_net.parameters():
                param.requires_grad = False
            
            # Disabled torch.compile to avoid 'Compiler: cl is not found' on Windows
            # # Если PyTorch 2.0+, используем compile для ускорения
            # try:
            #     if hasattr(torch, 'compile'):
            #         agent.policy_net = torch.compile(
            #             agent.policy_net,
            #             mode='reduce-overhead',  # для CPU лучший режим
            #             fullgraph=False
            #         )
            #         logger.info(f"  ✓ {agent_name}: torch.compile enabled")
            # except Exception as e:
            #     logger.warning(f"  ⚠ {agent_name}: torch.compile failed: {e}")
        
        logger.info("✅ CPU optimizations applied")
        
        logger.info("=" * 60)
        logger.info("✅ 2+2 ENSEMBLE READY FOR TRADING")
        logger.info("=" * 60)
    
    def _find_config_file(self, dir_path: Path):
        for file in dir_path.glob("*.py"):
            if "alpha" in file.name or "config" in file.name:
                return file
        return None
    
    def _load_py_config(self, path: Path):
        unique_module_name = f"config_{path.parent.name}_{path.stem}"
        spec = importlib.util.spec_from_file_location(unique_module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = mod
        spec.loader.exec_module(mod)
        return mod.cfg
    
    def _load_norm_stats(self, model_dir: Path):
        ns_path = model_dir / "norm_stats.json"
        if ns_path.exists():
            with open(ns_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"norm_stats.json missing in {model_dir}")
    
    def _create_agent_from_config(self, cfg):
        return D3QN_PER_Agent(
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
    
    def _load_weights(self, agent, path, name):
        try:
            agent.load_model(str(path))
            agent.policy_net.eval()
            logger.info(f"✅ {name} Agent loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name} Agent: {e}")
            raise e
    
    def feature_engineering(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # Волатильность
        dataframe['volatility_90m'] = (
            (dataframe['high'].rolling(90).max() - dataframe['low'].rolling(90).min()) /
            (dataframe['low'].rolling(90).min() + 1e-9)
        )
        
        # VWAP и объемы
        dataframe['vwap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['vwap']
        dataframe['num_trades'] = dataframe['volume']
        dataframe['taker_base'] = dataframe['volume'] * 0.5
        dataframe['taker_quote'] = dataframe['quote_volume'] * 0.5
        
        # Заполняем NaN нулями
        dataframe = dataframe.fillna(0.0)
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.feature_engineering(dataframe)
    
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
        
        if p <= FEE_BUF:
            d_eff = d0_val
        else:
            d_eff = d0_val - (p - FEE_BUF)
            d_eff = max(d_min_val, d_eff)
        
        return -d_eff
    
    def get_model_input(self, dataframe: DataFrame, pair: str, side: str, model_num: int):
        """
        Получение входных данных для конкретной модели
        model_num: 1 или 2 (для выбора правильных norm_stats)
        """
        # 1. Данные (5 каналов)
        opens = dataframe['open'].values
        highs = dataframe['high'].values
        lows = dataframe['low'].values
        closes = dataframe['close'].values
        volumes = dataframe['volume'].values
        
        # Log Returns
        eps = 1e-9
        def calc_log_returns(arr):
            changes = arr[1:] / (arr[:-1] + eps)
            return np.log(np.maximum(changes, eps))
        
        r_opens = calc_log_returns(opens)
        r_highs = calc_log_returns(highs)
        r_lows = calc_log_returns(lows)
        r_closes = calc_log_returns(closes)
        r_volumes = np.log(volumes[1:] + 1.0)
        
        # Stack (5, N-1)
        data = np.stack([r_opens, r_highs, r_lows, r_closes, r_volumes])
        
        # 2. Выбор norm_stats
        if side == "LONG" and model_num == 1:
            current_norm_stats = self.norm_stats_long_1
        elif side == "LONG" and model_num == 2:
            current_norm_stats = self.norm_stats_long_2
        elif side == "SHORT" and model_num == 1:
            current_norm_stats = self.norm_stats_short_1
        else:  # SHORT model 2
            current_norm_stats = self.norm_stats_short_2
        
        # Нормализация
        asset_name = pair.replace('/', '')
        stats = None
        
        if asset_name in current_norm_stats:
            stats = current_norm_stats[asset_name]
            
        if stats:
            means = np.array(stats["mean"])
            stds = np.array(stats["std"])
            if means.ndim == 1: means = means.reshape(-1, 1)
            if stds.ndim == 1: stds = stds.reshape(-1, 1)
            
            # Ensure we use only the first 5 channels if stats has more
            if means.shape[0] > 5:
                means = means[:5]
                stds = stds[:5]
                
            data = (data - means) / (stds + 1e-8)
        else:
            return None
        
        # ВАЖНО: Инверсия должна быть ПОСЛЕ нормализации, как в trading_environment.py
        if side == "SHORT":
            data = data * -1.0

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Sliding window
        if data.shape[1] < 90:
            return None
        
        windows = sliding_window_view(data, window_shape=90, axis=1)
        windows = windows.transpose(1, 0, 2)
        
        # 4. Flatten & Extra Features
        batch_size = windows.shape[0]
        flat_feats = windows.reshape(batch_size, -1)
        add_feats = np.zeros((batch_size, 4), dtype=np.float32)
        combined = np.concatenate([flat_feats, add_feats], axis=1)
        input_tensor = torch.FloatTensor(combined).to(self.device)
        
        return input_tensor
    
    def get_model_input_cached(self, dataframe: DataFrame, pair: str, side: str, model_num: int):
        """
        Кэширующая версия get_model_input
        Ключ кэша = (pair, last_candle_timestamp, side, model_num)
        """
        # Генерируем ключ кэша
        last_timestamp = dataframe.iloc[-1]['date'] if 'date' in dataframe.columns else dataframe.index[-1]
        cache_key = (pair, str(last_timestamp), side, model_num)
        
        # Проверяем кэш
        with self.cache_lock:
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
        
        # Если не в кэше, вычисляем
        tensor = self.get_model_input(dataframe, pair, side, model_num)
        
        # Сохраняем в кэш
        with self.cache_lock:
            # Ограничиваем размер кэша
            if len(self.feature_cache) >= self.cache_max_size:
                # Удаляем старейший элемент (FIFO)
                self.feature_cache.pop(next(iter(self.feature_cache)))
            
            self.feature_cache[cache_key] = tensor
        
        return tensor
    
    def _parallel_inference(self, tensors_and_agents):
        """
        Выполняет inference для нескольких моделей параллельно
        tensors_and_agents: [(tensor, agent, name), ...]
        """
        def single_inference(tensor, agent):
            with torch.no_grad():
                return agent.policy_net(tensor).cpu().numpy()
        
        # Запускаем все инференсы параллельно
        futures = []
        for tensor, agent, name in tensors_and_agents:
            future = self.executor.submit(single_inference, tensor, agent)
            futures.append((future, name))
        
        # Собираем результаты
        results = {}
        for future, name in futures:
            results[name] = future.result()
        
        return results

    def _apply_soft_voting(self, long_actions, short_actions):
        """
        МЯГКОЕ ГОЛОСОВАНИЕ (Soft Voting / OR logic with Veto):
        
        Логика (L_open - число лонгистов 'за', S_open - число шортистов 'за'):
        1. L_open >= 1 И S_open == 0 -> LONG (Хотя бы один лонгист за, шортисты молчат)
        2. S_open >= 1 И L_open == 0 -> SHORT (Хотя бы один шортист за, лонгисты молчат)
        3. L_open >= 1 И S_open >= 1 -> CONFLICT (Обе стороны хотят войти -> вето/конфликт -> ждать)
        4. L_open == 0 И S_open == 0 -> WAIT (Всем пофиг)
        """
        result = {
            'enter_long': 0,
            'enter_short': 0,
            'reason': 'no_signal'
        }
        
        # Подсчет действий
        long_entry_count = sum(1 for a in long_actions if a == 1)
        short_entry_count = sum(1 for a in short_actions if a == 1)
        
        if long_entry_count >= 1 and short_entry_count == 0:
            # Хотя бы одна long модель "за", обе short модели "против" или hold
            result['enter_long'] = 1
            result['reason'] = f'soft_long_{long_entry_count}/2_short_0/2'
        elif short_entry_count >= 1 and long_entry_count == 0:
            # Хотя бы одна short модель "за", обе long модели "против" или hold
            result['enter_short'] = 1
            result['reason'] = f'soft_short_{short_entry_count}/2_long_0/2'
        else:
            # Любой другой случай = нет входа
            if long_entry_count > 0 and short_entry_count > 0:
                result['reason'] = f'conflict_L{long_entry_count}/2_S{short_entry_count}/2'
        
        return result
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        if self.config.get('runmode') not in ['live', 'dry_run']:
            return True
        
        try:
            from freqtrade.persistence import Trade  # type: ignore
            trades = Trade.get_trades([Trade.is_open.is_(True)]).all()
            current_shorts = sum(1 for t in trades if t.is_short)
            current_longs = sum(1 for t in trades if not t.is_short)
            
            MAX_LONGS = 50
            MAX_SHORTS = 50
            
            if side == "long":
                if current_longs >= MAX_LONGS:
                    return False
            elif side == "short":
                if current_shorts >= MAX_SHORTS:
                    return False
        except Exception:
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
        # 1. Базовая защита
        if len(dataframe) < self.startup_candle_count:
            return dataframe
        
        # 2. Оптимизация инференса
        deep_inference = self.config.get('deep_inference', False)
        if self.config.get('runmode') in ['live', 'dry_run']:
            df_input = dataframe.iloc[-91:].copy()
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
        tensor_long_1 = self.get_model_input_cached(df_input, metadata['pair'], side="LONG", model_num=1)
        tensor_long_2 = self.get_model_input_cached(df_input, metadata['pair'], side="LONG", model_num=2)
        tensor_short_1 = self.get_model_input_cached(df_input, metadata['pair'], side="SHORT", model_num=1)
        tensor_short_2 = self.get_model_input_cached(df_input, metadata['pair'], side="SHORT", model_num=2)
        
        if None in [tensor_long_1, tensor_long_2, tensor_short_1, tensor_short_2]:
            return dataframe
        
        # 4. ПАРАЛЛЕЛЬНЫЙ INFERENCE для всех 4 моделей одновременно
        inference_tasks = [
            (tensor_long_1, self.long_1_agent, "long_1"),
            (tensor_long_2, self.long_2_agent, "long_2"),
            (tensor_short_1, self.short_1_agent, "short_1"),
            (tensor_short_2, self.short_2_agent, "short_2")
        ]
        
        q_values = self._parallel_inference(inference_tasks)
        
        # Распаковываем результаты
        q_long_1 = q_values["long_1"]
        q_long_2 = q_values["long_2"]
        q_short_1 = q_values["short_1"]
        q_short_2 = q_values["short_2"]
        
        # 5. Получение действий напрямую из Q-values (БЕЗ ПОРОГОВ)
        # argmax по Q-values дает действие: 0=hold, 1=entry
        action_long_1 = np.argmax(q_long_1, axis=1)
        action_long_2 = np.argmax(q_long_2, axis=1)
        action_short_1 = np.argmax(q_short_1, axis=1)
        action_short_2 = np.argmax(q_short_2, axis=1)

        # 6. Применяем строгое голосование для каждой свечи
        n_predictions = len(action_long_1)
        target_idx = slice(-n_predictions, None)
        
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        
        dataframe['enter_long'] = dataframe['enter_long'].astype(np.int8)
        dataframe['enter_short'] = dataframe['enter_short'].astype(np.int8)
        
        final_long_signals = np.zeros(n_predictions, dtype=np.int8)
        final_short_signals = np.zeros(n_predictions, dtype=np.int8)
        
        for i in range(n_predictions):
            # Собираем действия для текущей свечи
            long_actions = [action_long_1[i], action_long_2[i]]
            short_actions = [action_short_1[i], action_short_2[i]]
            
            # Применяем soft voting правила
            decision = self._apply_soft_voting(long_actions, short_actions)
            
            # Сбор статистики
            if any(long_actions) or any(short_actions):
                self.conflict_stats['total_signals'] += 1
                if 'conflict' in decision['reason']:
                    self.conflict_stats['conflicts'] += 1
                elif decision['enter_long']:
                    self.conflict_stats['long_entries'] += 1
                elif decision['enter_short']:
                    self.conflict_stats['short_entries'] += 1
            
            final_long_signals[i] = decision['enter_long']
            final_short_signals[i] = decision['enter_short']
            
            # Логирование
            if (decision['enter_long'] or decision['enter_short']) and i < 3:
                logger.info(f"📊 ENTRY SIGNAL: {decision['reason']}")
            elif 'conflict' in decision['reason'] and i < 5:
                logger.warning(f"⚠️ VOTING CONFLICT: {decision['reason']}")

        # 7. Вывод статистики
        if self.conflict_stats['total_signals'] > 0 and self.conflict_stats['total_signals'] % 1000 == 0:
            conflict_rate = self.conflict_stats['conflicts'] / self.conflict_stats['total_signals'] * 100
            logger.info(f"📈 VOTING STATS: Conflicts={conflict_rate:.1f}% | "
                        f"Long={self.conflict_stats['long_entries']} | Short={self.conflict_stats['short_entries']}")
        
        # Запись финальных сигналов
        dataframe.iloc[target_idx, dataframe.columns.get_loc('enter_long')] = final_long_signals
        dataframe.iloc[target_idx, dataframe.columns.get_loc('enter_short')] = final_short_signals
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe