import sys
import json
import logging
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch

# --- 1. НАСТРОЙКА ПУТЕЙ ---
strategy_file = Path(__file__).resolve()
project_root = strategy_file.parent.parent.parent # third_party/rl-trading-binance

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Freqtrade imports
try:
    from freqtrade.strategy import IStrategy
except ImportError:
    logging.getLogger(__name__).error("Could not import freqtrade.strategy")
    class IStrategy: pass

logger = logging.getLogger(__name__)

# Agent imports
try:
    from agent import D3QN_PER_Agent
    # Нам также нужен load_config, если он у вас есть в utils.py,
    # но чтобы не зависеть от utils, я напишу загрузчик конфига прямо тут.
except ImportError as e:
    logger.error(f"CRITICAL: Could not import D3QN Agent! Check path: {project_root}")
    raise e

class CustomD3QNStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01

    # --- FreqUI PLOT CONFIG ---
    plot_config = {
        'main_plot': {
            'vwap': {'color': 'blue'},
        },
        'subplots': {
            "Volatility": {
                'volatility_90m': {'color': 'orange'}
            },
        }
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.device = torch.device("cpu")
        self.project_root = Path(__file__).parent.parent.parent
        
        # --- ПУТИ К МОДЕЛЯМ ---
        # Папка SHORT модели (ведущая, оттуда берем конфиг)
        self.short_model_dir = self.project_root / "output/alpha_seed_404_v13_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260115_time_002944"
        self.short_model_pth = self.short_model_dir / "best.pth"
        
        # Папка LONG модели
        self.long_model_dir = self.project_root / "output/alpha_seed_404_v13_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260115_time_162618"
        self.long_model_pth = self.long_model_dir / "best.pth"
        
        # --- 1. ЗАГРУЗКА КОНФИГОВ (ИНДИВИДУАЛЬНО) ---
        
        # SHORT Config
        cfg_file_short = self._find_config_file(self.short_model_dir)
        if not cfg_file_short:
            raise FileNotFoundError(f"Config .py not found in {self.short_model_dir}")
        logger.info(f"Loading SHORT config from {cfg_file_short}")
        self.cfg_short = self._load_py_config(cfg_file_short)

        # LONG Config
        cfg_file_long = self._find_config_file(self.long_model_dir)
        if not cfg_file_long:
             raise FileNotFoundError(f"Config .py not found in {self.long_model_dir}")
        logger.info(f"Loading LONG config from {cfg_file_long}")
        self.cfg_long = self._load_py_config(cfg_file_long)
        
        # --- 2. ЗАГРУЗКА NORM_STATS (ИНДИВИДУАЛЬНО) ---
        
        # SHORT Stats
        ns_path_short = self.short_model_dir / "norm_stats.json"
        if ns_path_short.exists():
            with open(ns_path_short, 'r') as f:
                self.norm_stats_short = json.load(f)
        else:
            raise FileNotFoundError(f"norm_stats.json missing in {self.short_model_dir}")

        # LONG Stats
        ns_path_long = self.long_model_dir / "norm_stats.json"
        if ns_path_long.exists():
            with open(ns_path_long, 'r') as f:
                self.norm_stats_long = json.load(f)
        else:
             raise FileNotFoundError(f"norm_stats.json missing in {self.long_model_dir}")

        # --- 3. ИНИЦИАЛИЗАЦИЯ АГЕНТОВ (БЕЗ ЗАГЛУШЕК) ---
        # Используем соответствующие конфиги
        self.long_agent = self._create_agent_from_config(self.cfg_long)
        self.short_agent = self._create_agent_from_config(self.cfg_short)

        # --- 4. ЗАГРУЗКА ВЕСОВ ---
        self._load_weights(self.long_agent, self.long_model_pth, "LONG")
        self._load_weights(self.short_agent, self.short_model_pth, "SHORT")

    def _find_config_file(self, dir_path: Path):
        """Ищет файл конфигурации (alpha_*.py) в папке."""
        for file in dir_path.glob("*.py"):
            if "alpha" in file.name or "config" in file.name:
                return file
        return None

    def _load_py_config(self, path: Path):
        """Загружает Python-модуль как конфиг с уникальным именем."""
        # Генерируем уникальное имя модуля (папка_файл), чтобы избежать кеширования sys.modules
        unique_module_name = f"config_{path.parent.name}_{path.name}"
        
        spec = importlib.util.spec_from_file_location(unique_module_name, path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Cannot load config from {path}")
             
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = mod # Регистрируем
        spec.loader.exec_module(mod)
        # Возвращаем объект cfg из модуля
        return mod.cfg

    def _create_agent_from_config(self, cfg):
        """Создает агента, используя параметры из загруженного конфига."""
        return D3QN_PER_Agent(
            state_shape=cfg.seq.state_shape,       # (10, 90, 1)
            action_dim=cfg.market.num_actions,     # 3
            cnn_maps=cfg.model.cnn_maps,
            cnn_kernels=cfg.model.cnn_kernels,
            cnn_strides=cfg.model.cnn_strides,
            cnn_dilations=cfg.model.cnn_dilations,
            dense_val=cfg.model.dense_val,
            dense_adv=cfg.model.dense_adv,
            additional_feats=cfg.model.additional_feats,
            dropout_model=cfg.model.dropout_p,
            device=self.device,
            
            # --- ПАРАМЕТРЫ ОБУЧЕНИЯ (Теперь берем реальные!) ---
            gamma=cfg.rl.gamma,
            learning_rate=cfg.rl.lr,
            batch_size=cfg.rl.batch_size,
            buffer_size=cfg.per.buffer_size,
            
            # Параметры, которых нет в cfg.rl, берем из cfg.per / cfg.eps или хардкодим дефолты из конфига
            target_update_freq=cfg.rl.target_update_freq,
            train_start=cfg.rl.train_start,
            per_alpha=cfg.per.per_alpha,
            per_beta_start=cfg.per.per_beta_start,
            per_beta_frames=cfg.per.per_beta_frames,
            
            eps_start=cfg.eps.eps_start,
            eps_end=cfg.eps.eps_end,
            eps_frames=cfg.eps.eps_decay_frames,
            epsilon=0.0, # Для инференса ставим 0
            
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
        # Спайк-детектор
        dataframe['volatility_90m'] = (dataframe['high'].rolling(90).max() - dataframe['low'].rolling(90).min()) / dataframe['low'].rolling(90).min()
        
        # Аппроксимация (ВРЕМЕННАЯ, пока нет реальных данных)
        dataframe['vwap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['vwap']
        dataframe['num_trades'] = dataframe['volume'] 
        dataframe['taker_base'] = dataframe['volume'] * 0.5
        dataframe['taker_quote'] = dataframe['quote_volume'] * 0.5
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.feature_engineering(dataframe)

    def get_model_input(self, dataframe: DataFrame, pair: str, side: str):
        # 1. Данные (10 каналов, 90 свечей)
        df_slice = dataframe.iloc[-90:].copy()
        
        channel_data = [
            df_slice['open'].values, df_slice['high'].values, df_slice['low'].values, df_slice['close'].values, df_slice['volume'].values,
            df_slice['quote_volume'].values, df_slice['num_trades'].values, df_slice['taker_base'].values, df_slice['taker_quote'].values, df_slice['vwap'].values
        ]
        feats = np.stack(channel_data) # (10, 90)
        
        # 2. Нормализация
        symbol = pair.split('/')[0] + pair.split('/')[1].split(':')[0] 
        
        # Выбираем правильные статы
        if side == "LONG":
            current_norm_stats = self.norm_stats_long
        else:
            current_norm_stats = self.norm_stats_short

        if symbol in current_norm_stats:
            stats = current_norm_stats[symbol]
            means = np.array(stats.get('mean', stats.get('means'))).reshape(-1, 1)
            stds = np.array(stats.get('std', stats.get('stds'))).reshape(-1, 1)
            feats = (feats - means) / (stds + 1e-8)
        else:
            feats = (feats - np.mean(feats, axis=1, keepdims=True)) / (np.std(feats, axis=1, keepdims=True) + 1e-8)

        # 3. ПОДГОТОВКА ДЛЯ model.py (Flatten + Concat)
        # Модель ждет (Batch, 904).
        # Сначала плющим историю: (10, 90) -> (900,)
        # Важно: model.py делает view(batch, 10, 90). Это row-major.
        # np.flatten() по умолчанию тоже row-major (C-style). Всё совпадает.
        flat_feats = feats.flatten() # (900,)
        
        # Доп фичи (4 шт) - Заглушка (0,0,0,0) или реальные данные?
        # В model.py они идут в extra_part.
        # Пока заглушка.
        add_feats = np.zeros(4, dtype=np.float32)
        
        # Склеиваем ВМЕСТЕ: [900] + [4] = [904]
        combined = np.concatenate([flat_feats, add_feats])
        
        # Превращаем в тензор (Batch, 904)
        input_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
        
        return input_tensor

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if len(dataframe) < 90: return dataframe
        last_idx = dataframe.index[-1]

        if dataframe.iloc[-1]['volatility_90m'] < 0.015:
            return dataframe

        # Получаем тензоры для каждой стороны
        state_tensor_long = self.get_model_input(dataframe, metadata['pair'], side="LONG")
        state_tensor_short = self.get_model_input(dataframe, metadata['pair'], side="SHORT")
        
        with torch.no_grad():
            # Передаем ТОЛЬКО state_tensor (без второго аргумента)
            # D3QN_PER_Agent.policy_net -> DuelingQNetwork.forward(state)
            q_long = self.long_agent.policy_net(state_tensor_long)
            act_long = q_long.argmax(dim=1).item()
            
            q_short = self.short_agent.policy_net(state_tensor_short)
            act_short = q_short.argmax(dim=1).item()

        if act_long == 1: dataframe.loc[last_idx, 'enter_long'] = 1
        if act_short == 2: dataframe.loc[last_idx, 'enter_short'] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
