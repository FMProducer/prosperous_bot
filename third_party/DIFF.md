## 🔧 UNIFIED DIFF ПАТЧИ: profit_holding_bonus

Предлагаю **3 патча** для добавления нового параметра `cfg.market.profit_holding_bonus`, который будет мотивировать агента держать прибыльные позиции дольше.[1][2]

***

## 📋 ИЗМЕНЕНИЯ

| Файл | Локация | Изменение | Тип |
|------|---------|-----------|-----|
| **alpha_seed_404_v8.py** | ~Строка 178 | `cfg.market.profit_holding_bonus = 0.02` | Параметр конфигурации |
| **trading_environment.py** | ~Строка 52 | `profit_holding_bonus: float = 0.0` в `__init__` | Передача в класс |
| **trading_environment.py** | ~Строка 114 | `self.profit_holding_bonus = ...` | Сохранение атрибута |
| **trading_environment.py** | ~Строка 588 | Логика расчета бонуса | Реализация награды |

***

## ПАТЧ 1: Конфигурация (alpha_seed_404_v8.py)

```diff
--- a/third_party/rl-trading-binance/configs/alpha_seed_404_v8.py
+++ b/third_party/rl-trading-binance/configs/alpha_seed_404_v8.py
@@ -175,6 +175,9 @@ cfg.market.greed_penalty_multiplier = 0.1
 # Штраф за преждевременный выход (удержание < 5 шагов)
 cfg.market.premature_exit_penalty = 0.15
 
+# НОВАЯ НАГРАДА: Бонус за удержание прибыльной позиции (за каждый шаг в прибыли)
+cfg.market.profit_holding_bonus = 0.02
+
 # --- Thresholds for Shaped Rewards ---
 
 # Порог времени удержания для начала прогрессивного штрафа (шагов)
```

***

## ПАТЧ 2: Параметр в __init__ (trading_environment.py)

```diff
--- a/third_party/rl-trading-binance/trading_environment.py
+++ b/third_party/rl-trading-binance/trading_environment.py
@@ -49,6 +49,7 @@ class TradingEnvironment(gym.Env):
         holding_penalty_multiplier: float = 0.0,
         greed_penalty_multiplier: float = 0.0,
         premature_exit_penalty: float = 0.0,
+        profit_holding_bonus: float = 0.0,
         # --- Thresholds for Shaped Rewards ---
         holding_penalty_threshold: int = 15,
         greed_penalty_threshold: float = 0.50,
@@ -110,6 +111,7 @@ class TradingEnvironment(gym.Env):
         self.holding_penalty_multiplier = holding_penalty_multiplier
         self.greed_penalty_multiplier = greed_penalty_multiplier
         self.premature_exit_penalty = premature_exit_penalty
+        self.profit_holding_bonus = profit_holding_bonus
 
         # --- Thresholds ---
         self.holding_penalty_threshold = holding_penalty_threshold
```

***

## ПАТЧ 3: Логика в calculate_shaped_reward (trading_environment.py)

```diff
--- a/third_party/rl-trading-binance/trading_environment.py
+++ b/third_party/rl-trading-binance/trading_environment.py
@@ -585,6 +585,15 @@ class TradingEnvironment(gym.Env):
         if action == 3 and prev_position != 0:  # CLOSE position
             holding_duration = 0 if self.position_entry_step is None else (self.step_idx - self.position_entry_step)
 
+            # --- NEW: Profit Holding Bonus ---
+            # Reward agent for holding profitable positions
+            if self.profit_holding_bonus > 0 and trade_pnl > 0:
+                # Bonus is proportional to holding duration and trade profitability
+                holding_bonus = holding_duration * self.profit_holding_bonus
+                shaped_reward += holding_bonus
+                # Optional: scale by profit size
+                # holding_bonus = (holding_duration * self.profit_holding_bonus) * (trade_pnl / self.initial_balance)
+
             # --- 6. NEW: Perfect Entry Reward ---
             # Reward for profitable trades that never went into loss
             if self.perfect_entry_reward > 0 and trade_pnl > 0 and self.min_unrealized_pnl >= 0:
```

***

## 🎯 ЛОГИКА

### Когда применяется:
- При закрытии позиции (`action == 3`)
- Только для **прибыльных** сделок (`trade_pnl > 0`)
- Если `profit_holding_bonus > 0`

### Формула:
```python
holding_bonus = holding_duration × profit_holding_bonus
```

Где:
- `holding_duration` = количество шагов удержания позиции
- `profit_holding_bonus` = 0.02 (из конфига)

***

## 📊 ПРИМЕРЫ

| Holding (шагов) | Bonus | Trade PnL | Reward Bonus | Комментарий |
|-----------------|-------|-----------|--------------|-------------|
| 5 | 0.02 | +10 USDT | **+0.10** | Короткая сделка |
| 15 | 0.02 | +15 USDT | **+0.30** | Средняя сделка |
| 30 | 0.02 | +20 USDT | **+0.60** | Длинная сделка |
| 60 | 0.02 | +25 USDT | **+1.20** | Максимальная длительность |

***

## 💡 ЭФФЕКТ НА ОБУЧЕНИЕ

### 1. 📈 Мотивация держать прибыль дольше
- Прогрессивный бонус за каждый шаг удержания
- 30 шагов = +0.60 reward (эквивалент +6 USDT PnL)
- Компенсирует страх "жадности"

### 2. 🎯 Баланс с premature_exit_penalty

| Holding Duration | Penalty | Holding Bonus | Net Reward |
|------------------|---------|---------------|------------|
| 5 шагов | -0.15 | +0.10 | **-0.05** |
| 10 шагов | 0.00 | +0.20 | **+0.20** |
| 20 шагов | 0.00 | +0.40 | **+0.40** |

**Вывод:** Агент научится выжидать минимум 8-10 шагов для максимизации reward.[2]

### 3. ✂️ Только для прибыльных сделок
- Убыточные сделки **НЕ** получают бонус
- Дополнительно к `holding_penalty_multiplier` для убытков
- Четкий контраст: держать прибыль = выгодно, держать убыток = невыгодно

### 4. 🔄 Синергия с good_exit_bonus

**Сценарий:** Прибыль +20 USDT, удержание 25 шагов, выход на +18 USDT (90% от пика)

- `good_exit_bonus`: +0.30 (выход на 90%)
- `profit_holding_bonus`: +0.50 (25 × 0.02)
- **Total reward:** +0.80

Максимальная награда за терпеливое удержание **И** качественный выход.[1]

***