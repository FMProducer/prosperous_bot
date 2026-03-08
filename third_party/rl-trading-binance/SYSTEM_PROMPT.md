# RL Trading System Architecture

> **❗ Внимание**  
> Этот документ содержит **только архитектурные детали**, относящиеся к торговой логике.  
> Для информации о файлах и конфигах см. [README.md](README.md).

## 1. Core Framework
- **Execution Engine**: Freqtrade (inherit from `IStrategy`).  
- **Торговая логика**: Реализована в `user_data/strategies/CustomD3QNStrategy4z.py`.  
- **Единственный рабочий конфиг**: `user_data/config_rl4z.json` (не `configs/*.json`).  
- **Важно**: Все параметры, влияющие на торговлю, должны быть в этом файле.

## 2. Key Components
### 📌 Strategy Architecture (2+2 Ensemble)
- **4 модели**: 2 LONG-only + 2 SHORT-only (загружаются из `output/alpha_seed_*/saved_models/`).  
- **Входные данные**: 10 каналов (OHLCV + Volume, QuoteVolume, VWAP и др.).  
- **Нормализация**: Q-значения нормализуются через `q_min`/`q_max` из `config_rl4z.json`.  
- **Решение**:  
  ```python
  if normalized_advantage > dynamic_epsilon:  # Динамический порог
      vote = 1
  if opposite_side_votes > 0:  # Вето-механизм
      block_entry()
  ```

### ⚠️ Critical Constraints
1. **No Lookahead Bias**  
   - В бэктесте **запрещено** использовать данные из будущего (см. [analysis.md](analysis.md)).  
   - Пример: `dp.get_analyzed_dataframe` возвращает полный датасет → в бэктесте используйте только данные ≤ current_time.

2. **Backtest vs Live Separation**  
   - Всегда проверяйте режим через `self.config.get("runmode")`:  
     ```python
     if self.config.get("runmode") == "backtest":
         # Используйте безопасные методы
     else:
         # Ливе-логика
     ```

3. **Файлы вне `user_data/`**  
   - Все файлы в `third_party/rl-trading-binance/` (кроме документации) **не участвуют в торговле**.  
   - Не редактируйте их для изменения стратегии!

## 3. Dynamic Components
| Компонент | Как управляется | Где проверить |
|-----------|----------------|---------------|
| **Dynamic Epsilon** | `config_rl4z.json` → `rl_ensemble.enable_dynamic_epsilon` | Лог: `🛡️ DEFENSIVE MODE` |
| **Slot Allocation** | `config_rl4z.json` → `rl_ensemble.enable_dynamic_slots` | Лог: `🎰 SLOTS: Long=5 Short=3` |
| **Regime Filter** | `config_rl4z.json` → `use_regime_filter` | Лог: `📊 Regime: BULLISH` |

## 4. Testing & Validation
- **Запуск бэктеста**:  
  ```bash
  freqtrade backtest --config user_data/config_rl4z.json --strategy CustomD3QNStrategy4z
  ```
- **Ключевые проверки**:  
  1. Нет lookahead bias (анализ временных меток в логах)  
  2. Все модели загружены: `grep "Loaded model" logs/freqtrade.log`  
  3. Динамические параметры активны: `grep "Dynamic epsilon" logs/freqtrade.log`

## 📌 Ссылки на актуальную информацию
- [README.md](README.md) → Полная файловая структура и инструкции  
- [analysis.md](analysis.md) → Как избежать lookahead bias  
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → Решение типовых проблем  