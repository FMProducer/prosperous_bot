from datetime import datetime
from pandas import DataFrame
import numpy as np

try:
    from freqtrade.optimize.hyperopt import IHyperOptLoss  # type: ignore
except ImportError:
    class IHyperOptLoss: pass

class CustomTSLHyperopt(IHyperOptLoss):
    """
    Оптимизация параметров Trailing Stop Loss.
    Минимизируем: -Sharpe Ratio (чтобы максимизировать Sharpe).
    """
    
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs
    ) -> float:
        """
        Целевая функция: максимизация Sharpe Ratio.
        """
        if trade_count == 0:
            return 999999  # Штраф за отсутствие сделок
        
        # Расчет доходности и волатильности
        total_profit = results['profit_ratio'].sum()
        std_dev = results['profit_ratio'].std()
        
        if std_dev == 0:
            return 999999
        
        # Аннуализированный Sharpe Ratio (предполагаем 1-минутные бары)
        num_periods = (max_date - min_date).total_seconds() / 60
        annual_return = (total_profit / num_periods) * 525600  # минут в году
        annual_std = std_dev * np.sqrt(525600)
        
        sharpe_ratio = annual_return / annual_std if annual_std > 0 else -999
        
        # Штраф за превышение Max Drawdown > 20%
        # Рассчитываем Max Drawdown вручную по кривой доходности
        cumulative_profit = results.sort_values('close_date')['profit_ratio'].cumsum()
        max_profit = cumulative_profit.cummax()
        drawdowns = cumulative_profit - max_profit
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0
        
        penalty = 0
        if max_drawdown > 0.20:
            penalty = (max_drawdown - 0.20) * 10
        
        return -sharpe_ratio + penalty  # Минимизируем (инвертируем Sharpe)