# Progress Report: RL Trading System Optimization

## Completed Tasks
- [x] Make `entry_discount_pct` hyperoptable in `CustomD3QNStrategy4z.py`.
- [x] Refactor EMA filters: Global filter is now per-ticker on a higher timeframe (3m, 5m, 15m).
- [x] Maintain BTC EMA visualization in dashboards by keeping indicator names.
- [x] Update `informative_pairs` to support multi-timeframe data for all whitelist tickers.

### 30. Гибкий Alpha Decay и устранение логической блокировки
- **Status**: IMPLEMENTED
- **Logic**: 
    - **Разблокировка сигналов**: Из `populate_entry_trend` удалена проверка на наличие открытой позиции. Теперь `votes_long` and `votes_short` рассчитываются на каждой свече.
    - **Conditional Alpha Decay**: Логика выхода по сигналу полностью перенесена в `custom_exit`. Теперь выход по развороту тренда срабатывает **только при убытках** хуже `emergency_exit_threshold`.
    - **Прибыльные сделки**: Если сделка в плюсе, сигналы ансамбля игнорируются, и управление полностью передается кастомному TSL.
- **Result**: Достигнут идеальный баланс между защитой капитала при ошибке входа и максимизацией прибыли при верном прогнозе.

### 31. Калибровка Alpha Stop для Paper Trading
- **Status**: TESTING (Dry-run)
- **Parameter**: `emergency_exit_threshold` установлен на **-0.0575**.
- **Observation**: Выявлено, что порог начинает активно влиять на результаты начиная с -0.0574.
- **Expected Behavior**: Стратегия игнорирует встречные сигналы в зоне прибыли и малых убытков, активируя экстренный выход по сигналу только при просадке глубже 5.75%.

### 32. Исправление ATR Dynamic Floor (Institutional Stop)
- **Status**: IMPLEMENTED & FIXED
- **Fix**: Математика изменена с выбора самого узкого стопа на выбор более широкой дистанции между TSL и ATR (с ограничением по d0).
- **Parameters**: Добавлены `atr_multiplier` (текущий: **1.572**) и `atr_period` (24) для точной настройки под волатильность крипторынка.
- **Result**: Стоп-лосс теперь адаптивно расширяется во время рыночного шума, предотвращая преждевременное выбивание сделок.

### 🏆 Benchmark: Golden Backtest (Jan 2026)
- **Profit Factor**: 5.95
- **Win Rate**: 85.9% (55 Win / 9 Loss)
- **Drawdown**: 0.06% (Absolute)
- **Avg Duration**: 7 minutes
- **Alpha Stop Efficiency**: 5 trades saved with avg loss -1.88% (instead of hard stop).
- **Setup**: ROI Table + TSL + Alpha Stop (-0.0575) + ATR Floor (1.572).


## 🛠 Hyperopt Strategy (30+ Parameters)

Для предотвращения переобучения (overfitting) и "проклятия размерности", принята тактика **поэтапной групповой оптимизации**:

### Этап 1: Двигатель (Входы / Entry)
- **Space**: `--spaces buy`
- **Параметры**: `rl_epsilon_long/short`, `rl_long/short_threshold`, `min_quote_volume_usd`, `vol_f1/f2_...`
- **Цель**: Максимальный Profit Factor и Win Rate. Фиксируем стоп на -4% и отключаем TSL.

### Этап 2: Тормоза (Выходы / Exit & Alpha Stop)
- **Space**: `--spaces sell`
- **Параметры**: `stoploss`, `minimal_roi`, `emergency_exit_threshold`, `rl_exit_long/short_threshold`.
- **Цель**: Минимизация просадки (Drawdown). Поиск точки "испарения альфы".

### Этап 3: Турбо (Трейлинг / TSL)
- **Space**: `--spaces sell`
- **Параметры**: `d0`, `d_min`, `hysteresis`, `p_target`, `tsl_exponent`.
- **Цель**: Увеличение средней прибыли на сделку. Проводится при замороженных параметрах Этапа 1.

### Этап 4: Адаптация (Волатильность / ATR Floor)
- **Space**: `--spaces sell`
- **Параметры**: `atr_multiplier`, `atr_period`.
- **Цель**: Финальная доводка защиты от шума. `atr_period` рекомендуется держать в диапазоне 24-30 для 1m таймфрейма.

## Next Steps
- [ ] **Paper Trading Monitoring**: Оценка поведения ATR Floor с множителем 1.572.
- [ ] **Extended Backtest (Jan-Mar 2026)**: Confirm stability over a longer period with optimized parameters.
- [ ] **Dry-Run Monitoring (24h)**: Evaluate Maker-order mechanics and unfilled limit cancellations.
- [ ] **Epsilon Calibration**: Potentially lower `rl_epsilon_long` and `rl_epsilon_short` to increase trade frequency if needed.
- [ ] **Nonlinear TSL Efficiency**: Continuous monitoring of TSL performance in real-time execution.
