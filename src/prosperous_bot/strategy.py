# strategy.py
# Модуль стратегий для Trading AI Agent by FMProducer

import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Абстрактная стратегия.
    Все пользовательские стратегии должны реализовать метод generate_signals().
    """
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pass


class MLStrategy(BaseStrategy):
    """
    Стратегия на основе обученной ML-модели (например, XGBoost).
    """
    def __init__(self, model):
        self.model = model

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(df), index=df.index).astype(bool)


class RuleBasedStrategy(BaseStrategy):
    """
    Стратегия на основе пользовательских технических условий.
    """
    def __init__(self, signal_func, config, last_signal):
        self.signal_func = signal_func
        self.config = config
        self.last_signal = last_signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return self.signal_func(df, self.config, self.last_signal)


class HybridStrategy(BaseStrategy):
    """
    Гибридная стратегия: логическое пересечение сигналов от ML и rule-based стратегий.
    """
    def __init__(self, ml_strategy: MLStrategy, rule_strategy: RuleBasedStrategy):
        self.ml = ml_strategy
        self.rule = rule_strategy

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ml_signals = self.ml.generate_signals(df)
        rule_signals = self.rule.generate_signals(df)
        return ml_signals & rule_signals
