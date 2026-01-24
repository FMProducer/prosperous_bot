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
from typing import Dict, Optional, List, Any


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
    d_min = DecimalParameter(0.001, 0.05, default=0.01, space='stoploss', load=True)
    hysteresis = DecimalParameter(0.001, 0.02, default=0.002, space='stoploss', load=True)
    
    plot_config = {
        'main_plot': {},
        'subplots': {}
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Принудительно включаем шорты
        self.can_short = True
        
        # === CPU ОПТИМИЗАЦИИ ===
        # 1. Установить количество потоков для PyTorch
        num_cpu_threads = config.get('cpu_threads', 4)  # по умолчанию 4 потока
        torch.set_num_threads(num_cpu_threads)
        torch.set_num_interop_threads(num_cpu_threads)
        
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
        self.long_1_norm_stats_path = self.long_1_model_dir / "norm_stats.json"
        
        # Long Model 2: A2C mean-reversion (используем ту же модель для примера, замените на вашу вторую)
        self.long_2_model_dir = self.project_root / "output/alpha_seed_406_ohlcv_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260124_time_061330"
        self.long_2_model_pth = self.long_2_model_dir / "best.pth"
        self.long_2_norm_stats_path = self.long_2_model_dir / "norm_stats.json"
        
        # Short Model 1: SAC bearish trending
        self.short_1_model_dir = self.project_root / "output/alpha_seed_404_ohlcv_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260118_time_225844"
        self.short_1_model_pth = self.short_1_model_dir / "best.pth"
        self.short_1_norm_stats_path = self.short_1_model_dir / "norm_stats.json"
        
        # Short Model 2: PPO short mean-reversion (используем ту же модель для примера, замените на вашу вторую)
        self.short_2_model_dir = self.project_root / "output/alpha_seed_406_ohlcv_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260123_time_180141"
        self.short_2_model_pth = self.short_2_model_dir / "best.pth"
        self.short_2_norm_stats_path = self.short_2_model_dir / "norm_stats.json"
        
        # --- ЗАГРУЗКА КОНФИГОВ ---
        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING 2+2 ENSEMBLE SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Project Root: {self.project_root}")
        
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
        self.norm_stats_long_1 = self._load_norm_stats(self.long_1_norm_stats_path)
        self.norm_stats_long_2 = self._load_norm_stats(self.long_2_norm_stats_path)
        self.norm_stats_short_1 = self._load_norm_stats(self.short_1_norm_stats_path)
        self.norm_stats_short_2 = self._load_norm_stats(self.short_2_norm_stats_path)
        
        # --- ОПРЕДЕЛЕНИЕ РЕЖИМА MIRROR MODE ---
        # ЖЕСТКО ЗАДАЕМ TRUE, так как модели обучены на зеркальном графике.
        self.short_1_is_mirror = True
        self.short_2_is_mirror = True
        
        logger.info(f"ℹ️ SHORT_1 Mirror Mode: {self.short_1_is_mirror} (Hardcoded)")
        logger.info(f"ℹ️ SHORT_2 Mirror Mode: {self.short_2_is_mirror} (Hardcoded)")
        
        # --- НАСТРОЙКИ ГОЛОСОВАНИЯ (из конфига) ---
        self.vote_threshold_long = config.get('rl_long_threshold', 2)
        self.vote_threshold_short = config.get('rl_short_threshold', 2)
        self.enable_veto = config.get('rl_enable_veto', True)
        
        logger.info(f"🗳️ Voting Rules: Long>={self.vote_threshold_long}, Short>={self.vote_threshold_short}, Veto={self.enable_veto}")

        # --- ИНИЦИАЛИЗАЦИЯ 4 АГЕНТОВ ---
        logger.info("📦 Creating agents...")
        self.long_1_agent = self._create_agent_from_config(self.cfg_long_1, mirror_mode=False)
        self.long_2_agent = self._create_agent_from_config(self.cfg_long_2, mirror_mode=False)
        self.short_1_agent = self._create_agent_from_config(self.cfg_short_1, mirror_mode=self.short_1_is_mirror)
        self.short_2_agent = self._create_agent_from_config(self.cfg_short_2, mirror_mode=self.short_2_is_mirror)
        
        # --- ЗАГРУЗКА ВЕСОВ ---
        # Safety check: Ensure Long and Short models are not pointing to the same file
        if self.long_1_model_pth == self.short_1_model_pth:
            logger.error("🚨 CRITICAL: LONG_1 and SHORT_1 model paths are IDENTICAL! Check paths.")
        if self.long_2_model_pth == self.short_2_model_pth:
            logger.error("🚨 CRITICAL: LONG_2 and SHORT_2 model paths are IDENTICAL! Check paths.")

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
        
        if not self.can_short:
            logger.warning("⚠️ WARNING: can_short is False! Short signals will be ignored.")

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
    
    def _load_norm_stats(self, ns_path: Path):
        if ns_path.exists():
            logger.info(f"Loading norm_stats from {ns_path}")
            with open(ns_path, 'r') as f:
                stats = json.load(f)
                # Basic validation to ensure file is not empty or malformed
                if not stats:
                    logger.warning(f"⚠️ WARNING: norm_stats at {ns_path} is empty!")
                return stats
        else:
            raise FileNotFoundError(f"norm_stats.json missing at {ns_path}")
    
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

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str, model_num: int, asset_name: str) -> Optional[torch.Tensor]:
        # 1. Данные (5 каналов)
        # FIX: Calculate Log Returns and Log Volume to match training data!
        # Raw prices (e.g. 60000) vs Log Returns (e.g. 0.001) caused the model to fail.
        opens = dataframe['open'].values
        highs = dataframe['high'].values
        lows = dataframe['low'].values
        closes = dataframe['close'].values
        volumes = dataframe['volume'].values
        
        eps = 1e-9
        # returns are (t) / (t-1). Result length is N-1.
        r_opens = np.log(np.maximum(opens[1:] / (opens[:-1] + eps), eps))
        r_highs = np.log(np.maximum(highs[1:] / (highs[:-1] + eps), eps))
        r_lows = np.log(np.maximum(lows[1:] / (lows[:-1] + eps), eps))
        r_closes = np.log(np.maximum(closes[1:] / (closes[:-1] + eps), eps))
        r_volumes = np.log(volumes[1:] + 1.0) # Align length with returns
        
        data = np.stack([r_opens, r_highs, r_lows, r_closes, r_volumes]).astype(np.float32) # (5, N-1)

        # 2. Выбор norm_stats
        if side == "LONG":
            current_norm_stats = self.norm_stats_long_1 if model_num == 1 else self.norm_stats_long_2
        else: # SHORT
            current_norm_stats = self.norm_stats_short_1 if model_num == 1 else self.norm_stats_short_2
            
        if not current_norm_stats:
            logger.error(f"Norm stats missing for model {model_num} {side}")
            return None

        # Нормализация
        if asset_name not in current_norm_stats:
            # Fallback logic
            alt_name = pair.split(':')[0]
            if alt_name in current_norm_stats:
                asset_name = alt_name
            else:
                # Uncomment for debugging if signals are missing
                # logger.warning(f"Missing norm_stats for {asset_name} in model {model_num} {side}")
                return None

        stats = current_norm_stats[asset_name]
        
        # Векторизованная нормализация
        means = np.array(stats["mean"][:5], dtype=np.float32).reshape(5, 1)
        stds = np.array(stats["std"][:5], dtype=np.float32).reshape(5, 1)
        
        data = (data - means) / (stds + 1e-8)
        
        # Inversion logic
        should_invert = False
        if side == "SHORT":
            if (model_num == 1 and self.short_1_is_mirror) or (model_num == 2 and self.short_2_is_mirror):
                should_invert = True
        
        if should_invert:
            # Revert: Invert ALL channels (including Volume) to match TradingEnvironment training logic
            data = data * -1.0

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Sliding window
        if data.shape[1] < 90:
            return None
        
        windows = sliding_window_view(data, window_shape=90, axis=1)
        # FIX: Transpose to (Batch, 90, 5) to match TradingEnvironment's Time-major flattening
        windows = windows.transpose(1, 2, 0)
        
        # 4. Flatten & Extra Features
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

    def _apply_soft_voting(
        self, 
        long_actions: List[int], 
        short_actions: List[int], 
        has_long: bool, 
        has_short: bool
    ) -> Dict[str, Any]:

        # --- Корректный подсчет голосов с учетом mirror_mode ---
        # Для LONG моделей, голос "ЗА" - это всегда действие 1.
        l_votes = list(long_actions).count(1)

        # Для SHORT моделей, голос "ЗА" зависит от режима.
        s_votes = 0
        # Модель 1
        if self.short_1_is_mirror:
            if short_actions[0] == 1: s_votes += 1  # В зеркальном режиме "лонг" (1) означает шорт.
        else:
            if short_actions[0] == 2: s_votes += 1  # В обычном режиме "шорт" - это действие 2.
        # Модель 2
        if self.short_2_is_mirror:
            if short_actions[1] == 1: s_votes += 1
        else:
            if short_actions[1] == 2: s_votes += 1

        # Проверка по порогу (Threshold) из конфига
        l_signal = (l_votes >= self.vote_threshold_long and l_votes > 0)
        s_signal = (s_votes >= self.vote_threshold_short and s_votes > 0)

        res = {
            'enter_long': 0, 
            'enter_short': 0, 
            'reason': f"L_votes:{l_votes} (Need {self.vote_threshold_long}) S_votes:{s_votes} (Need {self.vote_threshold_short})"
        }

        if has_long or has_short:
            res['reason'] += " | Position exists"
            return res

        # Вето, если обе стороны дали сигнал и вето включено
        if self.enable_veto and l_signal and s_signal:
            res['reason'] += " | Veto: Conflict"
            return res

        if l_signal:
            res['enter_long'] = 1
            res['reason'] += " | LONG Signal"
        elif s_signal and self.can_short:
            res['enter_short'] = 1
            res['reason'] += " | SHORT Signal"
        else:
            res['reason'] += " | No Consensus"

        return res

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
        asset_name = metadata['pair'].split(':')[0].replace('/', '')
        tensor_long_1 = self.get_model_input_cached(df_input, metadata['pair'], side="LONG", model_num=1, asset_name=asset_name)
        tensor_long_2 = self.get_model_input_cached(df_input, metadata['pair'], side="LONG", model_num=2, asset_name=asset_name)
        tensor_short_1 = self.get_model_input_cached(df_input, metadata['pair'], side="SHORT", model_num=1, asset_name=asset_name)
        tensor_short_2 = self.get_model_input_cached(df_input, metadata['pair'], side="SHORT", model_num=2, asset_name=asset_name)
        
        if None in [tensor_long_1, tensor_long_2, tensor_short_1, tensor_short_2]:
            # Логируем причину пропуска (опционально, можно закомментировать)
            # if tensor_long_1 is None: logger.warning(f"Missing input for LONG_1 on {metadata['pair']}")
            # if tensor_long_2 is None: logger.warning(f"Missing input for LONG_2 on {metadata['pair']}")
            # if tensor_short_1 is None: logger.warning(f"Missing input for SHORT_1 on {metadata['pair']}")
            # if tensor_short_2 is None: logger.warning(f"Missing input for SHORT_2 on {metadata['pair']}")
            return dataframe
        
        # Sanity Check: Ensure Short tensor is not identical to Long tensor (should be inverted)
        if tensor_long_1 is not None and tensor_short_1 is not None:
            if torch.equal(tensor_long_1, tensor_short_1):
                logger.warning(f"🚨 CRITICAL: LONG_1 and SHORT_1 tensors are IDENTICAL for {metadata['pair']}! Inversion failed?")

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

        # DEBUG: Log action distribution to verify models are outputting signals
        if self.config.get('runmode') not in ['live', 'dry_run']:
            u_s1, c_s1 = np.unique(action_short_1, return_counts=True)
            u_s2, c_s2 = np.unique(action_short_2, return_counts=True)
            logger.info(f"DEBUG {metadata['pair']} Short Actions: S1={dict(zip(u_s1, c_s1))} S2={dict(zip(u_s2, c_s2))}")

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
            l1 = int(action_long_1[i])
            l2 = int(action_long_2[i])
            s1 = int(action_short_1[i])
            s2 = int(action_short_2[i])
            
            long_actions = [l1, l2]
            short_actions = [s1, s2]
            
            # Применяем soft voting правила
            decision = self._apply_soft_voting(
                long_actions, 
                short_actions,
                has_long=has_long,
                has_short=has_short
            )
            
            # Сбор статистики
            if any(long_actions) or any(short_actions):
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
                # Преобразуем сырые действия в голоса (1=ЗА, 0=ПРОТИВ/ЖДАТЬ) для лога
                l_votes_log = [1 if a == 1 else 0 for a in long_actions]
                
                s_votes_log = []
                # Short 1
                if self.short_1_is_mirror:
                    s_votes_log.append(1 if short_actions[0] == 1 else 0)
                else:
                    s_votes_log.append(1 if short_actions[0] == 2 else 0)
                # Short 2
                if self.short_2_is_mirror:
                    s_votes_log.append(1 if short_actions[1] == 1 else 0)
                else:
                    s_votes_log.append(1 if short_actions[1] == 2 else 0)

                logger.info(f"{metadata['pair']} Candle {i} | LONG: {l_votes_log} | SHORT: {s_votes_log}")
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
        """Обязательный метод для торговли фьючерсами (шорт)"""
        return proposed_leverage