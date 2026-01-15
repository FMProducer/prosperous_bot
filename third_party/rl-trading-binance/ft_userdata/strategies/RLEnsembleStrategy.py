# RLEnsembleStrategy.py
import numpy as np
import pandas as pd
from pandas import DataFrame
try:
    from freqtrade.strategy import IStrategy  # type: ignore
except ImportError:
    class IStrategy:  # type: ignore
        def __init__(self, config: dict) -> None: pass

from stable_baselines3 import PPO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RLEnsembleStrategy(IStrategy):
    # Версия интерфейса (обязательно 3)
    INTERFACE_VERSION = 3
    
    # Таймфрейм (минутки, как при обучении)
    timeframe = '1m'
    
    # Включаем шорты
    can_short = True
    
    # Риск-менеджмент: отдаем контроль моделям, но страхуем трейлингом
    minimal_roi = {"0": 100}  # Выход только по сигналу
    stoploss = -0.99          # Стоп-лосс отключен (используем трейлинг)
    
    # Трейлинг-стоп (подстраховка от биржи)
    trailing_stop = True
    trailing_stop_positive = 0.005       # Активация при +0.5% профита
    trailing_stop_positive_offset = 0.01 # Держать дистанцию 1%
    trailing_only_offset_is_reached = True 

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # --- ПУТИ К МОДЕЛЯМ ---
        # Мы находимся в ft_userdata/strategies/
        # Нам нужно попасть в rl-trading-binance/output/
        # Поднимаемся: strategies -> ft_userdata -> rl-trading-binance -> output
        
        # Получаем абсолютный путь к папке стратегии
        strategy_dir = Path(__file__).parent
        
        # Вычисляем путь к output (корректируйте, если модели лежат глубже)
        # Пример: C:\Python\Prosperous_Bot\third_party\rl-trading-binance\output
        project_root = strategy_dir.parent.parent 
        models_dir = project_root / "output"
        
        self.long_model_path = models_dir / "long_model_best.zip"
        self.short_model_path = models_dir / "short_model_best.zip"
        
        self.long_model = None
        self.short_model = None
        
        logger.info(f"Looking for models in: {models_dir}")
        
        # Загрузка моделей (ленивая или сразу)
        try:
            self.long_model = PPO.load(self.long_model_path, device='cpu')
            self.short_model = PPO.load(self.short_model_path, device='cpu')
            logger.info("RL Models loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load models! Error: {e}")
            logger.error(f"Check paths: {self.long_model_path}")

    def feature_engineering(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        Расчет индикаторов (Ступень 1: Спайк-детектор)
        """
        # Rolling High/Low за 90 минут
        roll_max = dataframe['high'].rolling(90).max()
        roll_min = dataframe['low'].rolling(90).min()
        
        # Волатильность (Spike Factor)
        dataframe['volatility_90m'] = (roll_max - roll_min) / roll_min
        
        # --- ВАЖНО: ПРЕПРОЦЕССИНГ ДЛЯ МОДЕЛИ ---
        # Здесь вы должны добавить нормализацию или индикаторы, 
        # которые использовались при обучении (RSI, MACD и т.д.)
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14) 
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize signal columns to prevent KeyErrors
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return self.feature_engineering(dataframe)

    def get_model_input(self, dataframe: DataFrame):
        """
        Превращает DF в тензор. 
        CRITICAL: Порядок колонок должен быть 1-в-1 как в train.py!
        """
        last_90 = dataframe.iloc[-90:].copy()
        
        # Пример фичей (замените на свои!)
        # Если вы обучали на [Open, High, Low, Close, Volume], то:
        features = last_90[['open', 'high', 'low', 'close', 'volume']].values
        
        # Если нужна нормализация (например, деление на цену открытия окна):
        # start_price = features[0, 3] # Close первой свечи
        # features = features / start_price
        
        return features

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Логика входа (Ступень 2)
        Note: This logic calculates signals only for the last candle.
        It is designed for Live/Dry Run and will not generate trades in Backtesting.
        """
        if len(dataframe) < 90: return dataframe
        
        last_idx = dataframe.index[-1]
        
        # --- GATEKEEPER ---
        # Если волатильность < 1.5%, выходим. Экономим CPU.
        if dataframe.iloc[-1]['volatility_90m'] < 0.015:
            return dataframe

        # --- INFERENCE ---
        if self.long_model and self.short_model:
            input_data = self.get_model_input(dataframe)
            
            # Predict
            act_long, _ = self.long_model.predict(input_data, deterministic=True)
            act_short, _ = self.short_model.predict(input_data, deterministic=True)
            
            # Вход
            if act_long == 1:
                dataframe.loc[last_idx, 'enter_long'] = 1
            
            if act_short == 1:
                dataframe.loc[last_idx, 'enter_short'] = 1
                
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Выход по трейлингу или сигналу модели (если есть)
        return dataframe
