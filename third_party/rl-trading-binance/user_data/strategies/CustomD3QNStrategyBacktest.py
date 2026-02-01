# CustomD3QNStrategyBacktest.py
import sys
import json
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from pathlib import Path
from tqdm import tqdm  # Для прогресс-бара, если установлен

# --- Freqtrade Imports ---
try:
    from freqtrade.strategy import IStrategy  # type: ignore
    from freqtrade.persistence import Trade  # type: ignore
except ImportError:
    class IStrategy: pass
    class Trade: pass

from datetime import datetime
import importlib.util

# --- Logger Setup ---
logger = logging.getLogger(__name__)

class CustomD3QNStrategyBacktest(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_long = True
    can_short = True
    
    # --- Стратегические параметры ---
    minimal_roi = {"0": 100}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    
    # TSL Параметры (Hardcoded from original)
    d0 = 0.075
    d_min = 0.01
    hysteresis = 0.0001
    FEE_BUF = 0.0008 
    
    # Пороги входа
    THRESHOLD_LONG = 0.00114
    THRESHOLD_SHORT = 0.00114

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.device = torch.device("cpu") # Бэктест лучше на CPU или CUDA если есть
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.tsl_memory = {}
        
        # --- FIX: ДОБАВЛЯЕМ ROOT В PATH ПЕРЕД ЛЮБЫМИ ИМПОРТАМИ МОДЕЛЕЙ ---
        # Это позволяет файлам alpha_*.py делать "from config import cfg"
        if str(self.project_root) not in sys.path:
            sys.path.append(str(self.project_root))
        
        # --- ПУТИ (Скопировано из вашего файла) ---
        self.short_model_dir = self.project_root / "output/alpha_seed_404_v13_SHORT_ONLY/saved_models/rl_binance_futures_trading_date_20260115_time_002944"
        self.short_model_pth = self.short_model_dir / "best.pth"
        
        self.long_model_dir = self.project_root / "output/alpha_seed_404_v13_LONG_ONLY/saved_models/rl_binance_futures_trading_date_20260115_time_162618"
        self.long_model_pth = self.long_model_dir / "best.pth"

        # --- ЗАГРУЗКА ВСЕГО НЕОБХОДИМОГО ---
        # 1. Configs
        self.cfg_short = self._load_py_config(self._find_config_file(self.short_model_dir))
        self.cfg_long = self._load_py_config(self._find_config_file(self.long_model_dir))
        
        # 2. Stats
        with open(self.short_model_dir / "norm_stats.json", 'r') as f:
            self.norm_stats_short = json.load(f)
        with open(self.long_model_dir / "norm_stats.json", 'r') as f:
            self.norm_stats_long = json.load(f)
            
        # 3. Agents (Imports must work!)
        sys.path.append(str(self.project_root))
        try:
            from agent import D3QN_PER_Agent
            self.D3QN_PER_Agent = D3QN_PER_Agent # Сохраняем класс
        except ImportError:
            logger.error("Could not import Agent!")
            raise

        self.long_agent = self._create_agent(self.cfg_long)
        self.short_agent = self._create_agent(self.cfg_short)
        
        self._load_weights(self.long_agent, self.long_model_pth, "LONG")
        self._load_weights(self.short_agent, self.short_model_pth, "SHORT")

    # --- Helpers (Copy-Paste + Refactor) ---
    def _find_config_file(self, dir_path: Path):
        for file in dir_path.glob("*.py"):
            if "alpha" in file.name or "config" in file.name: return file
        return None

    def _load_py_config(self, path: Path):
        unique_name = f"config_bt_{path.parent.name}_{path.name}"
        spec = importlib.util.spec_from_file_location(unique_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)
        return mod.cfg

    def _create_agent(self, cfg):
        return self.D3QN_PER_Agent(
            state_shape=cfg.seq.state_shape,
            action_dim=cfg.market.num_actions,
            cnn_maps=cfg.model.cnn_maps,
            cnn_kernels=cfg.model.cnn_kernels,
            cnn_strides=cfg.model.cnn_strides,
            cnn_dilations=cfg.model.cnn_dilations,
            dense_val=cfg.model.dense_val,
            dense_adv=cfg.model.dense_adv,
            additional_feats=cfg.model.additional_feats,
            dropout_model=0.0, # No dropout in eval
            device=self.device,
            gamma=cfg.rl.gamma, learning_rate=cfg.rl.lr, batch_size=cfg.rl.batch_size, buffer_size=1000,
            target_update_freq=100, train_start=1000, per_alpha=0.6, per_beta_start=0.4,
            per_beta_frames=10000, eps_start=0.0, eps_end=0.0, eps_frames=1, epsilon=0.0, max_gradient_norm=1.0
        )

    def _load_weights(self, agent, path, name):
        agent.load_model(str(path))
        agent.policy_net.eval()
        logger.info(f"{name} Loaded.")

    # --- Feature Engineering ---
    def feature_engineering(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe['volatility_90m'] = (dataframe['high'].rolling(90).max() - dataframe['low'].rolling(90).min()) / dataframe['low'].rolling(90).min()
        dataframe['vwap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['quote_volume'] = dataframe['volume'] * dataframe['vwap']
        dataframe['num_trades'] = dataframe['volume'] # Mock
        dataframe['taker_base'] = dataframe['volume'] * 0.5 # Mock
        dataframe['taker_quote'] = dataframe['quote_volume'] * 0.5 # Mock
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.feature_engineering(dataframe)

    # --- LOGIC FOR ONE STEP (Refactored for Loop) ---
    def get_model_input_slice(self, df_slice: DataFrame, pair: str, side: str):
        # Аналог get_model_input, но принимает готовый срез (90 строк)
        channel_data = [
            df_slice['open'].values, df_slice['high'].values, df_slice['low'].values, 
            df_slice['close'].values, df_slice['volume'].values, df_slice['quote_volume'].values, 
            df_slice['num_trades'].values, df_slice['taker_base'].values, 
            df_slice['taker_quote'].values, df_slice['vwap'].values
        ]
        feats = np.stack(channel_data) # (10, 90)
        
        if side == "SHORT":
            feats = feats * -1.0
            
        symbol = pair.split('/')[0]
        stats = self.norm_stats_long[symbol] if side == "LONG" and symbol in self.norm_stats_long else \
                (self.norm_stats_short[symbol] if symbol in self.norm_stats_short else None)
        
        if stats:
            means = np.array(stats.get('mean', stats.get('means'))).reshape(-1, 1)
            stds = np.array(stats.get('std', stats.get('stds'))).reshape(-1, 1)
            feats = (feats - means) / (stds + 1e-8)
        else:
            feats = (feats - np.mean(feats, axis=1, keepdims=True)) / (np.std(feats, axis=1, keepdims=True) + 1e-8)
            
        flat_feats = feats.flatten()
        add_feats = np.zeros(4, dtype=np.float32)
        combined = np.concatenate([flat_feats, add_feats])
        return torch.FloatTensor(combined).unsqueeze(0).to(self.device)

    # --- MAIN BACKTEST LOOP ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Инициализируем колонки
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if len(dataframe) < 100: return dataframe
        
        logger.info(f"Starting RL Inference Loop for {metadata['pair']} ({len(dataframe)} candles)...")
        
        # Перебираем датафрейм. Начинаем с 90-го индекса, чтобы было окно
        # Используем tqdm для прогресса (если есть), или range
        iterator = range(90, len(dataframe))
        
        net_long = self.long_agent.policy_net
        net_short = self.short_agent.policy_net
        
        with torch.no_grad():
            for i in iterator:
                # 1. Проверка волатильности
                if dataframe.iloc[i]['volatility_90m'] < 0.015:
                    continue 
                
                # 2. Формируем срез
                # i+1, так как правая граница не включается. Срез [0:90] -> индексы 0..89 (90 шт)
                # Если i=90, срез [1:91] -> индексы 1..90 (90 шт)
                df_slice = dataframe.iloc[i-89 : i+1] 
                
                # Критическая проверка длины
                if len(df_slice) != 90:
                    continue
                
                # 3. Инференс с проверкой
                try:
                    state_long = self.get_model_input_slice(df_slice, metadata['pair'], "LONG")
                    state_short = self.get_model_input_slice(df_slice, metadata['pair'], "SHORT")
                    
                    # Проверка размерности тензора
                    if state_long.shape[0] == 0:
                        logger.warning(f"Empty tensor at index {i}")
                        continue

                    q_long_tensor = net_long(state_long)
                    q_short_tensor = net_short(state_short)
                    
                    # Безопасное извлечение
                    q_long = q_long_tensor.cpu().numpy()
                    q_short = q_short_tensor.cpu().numpy()
                    
                    if len(q_long) == 0 or len(q_short) == 0:
                         continue

                    q_long = q_long[0]
                    q_short = q_short[0]

                    # 4. Логика сигналов
                    idx = dataframe.index[i]
                    
                    # Расчет Advantage (Buy - Hold)
                    adv_long = q_long[1] - q_long[0]
                    adv_short = q_short[1] - q_short[0]

                    # ЛОГИРОВАНИЕ ДИАПАЗОНА (Для отладки)
                    if i % 1000 == 0:
                        logger.info(f"Step {i} STATS: Adv_Long={adv_long:.6f}, Adv_Short={adv_short:.6f}")

                    if adv_long > self.THRESHOLD_LONG:
                        logger.info(f"🔥 BUY SIGNAL at {i} (Vol: {dataframe.iloc[i]['volatility_90m']:.4f})")
                        dataframe.at[idx, 'enter_long'] = 1
                        
                    if adv_short > self.THRESHOLD_SHORT:
                        logger.info(f"📉 SELL SIGNAL at {i}")
                        dataframe.at[idx, 'enter_short'] = 1
                        
                except Exception as e:
                    # Логируем, но не падаем, чтобы дойти до конца
                    logger.warning(f"Error at index {i} for {metadata['pair']}: {e}")
                    continue
                    
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    # --- Custom Stoploss & Exit (Без изменений) ---
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        p = current_profit
        if trade.id not in self.tsl_memory:
            self.tsl_memory[trade.id] = -999.0
        last_p = self.tsl_memory[trade.id]
        
        if p >= (last_p + self.hysteresis):
            self.tsl_memory[trade.id] = p
            
        # Используем запомненное значение для ступенчатого эффекта
        calc_p = self.tsl_memory[trade.id]

        if calc_p <= self.FEE_BUF:
            d_eff = self.d0
        else:
            d_eff = self.d0 - (calc_p - self.FEE_BUF)
            d_eff = max(self.d_min, d_eff)
            
        return -d_eff

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        if trade.open_date_utc:
            duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
            if duration_min >= 60:
                return "timeout_60m"
        return None
