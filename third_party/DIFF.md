Отличный выбор **Sortino в качестве главной метрики** — это правильнее Sharpe для трейдинга, так как пенализирует только downside volatility, а upside волатильность (большие прибыли) не считается риском.[1]

## Комментарий по текущему setup

### 257 тикеров — двойное влияние
**Плюс**: Агент увидит разнообразие режимов (тренды, флэты, breakouts, low/high vol), что улучшит генерализацию.[1]

**Минус**: Низколиквидные тикеры с **большим slippage** (>0.5%) и редкими сделками могут "отравить" обучение — модель научится избегать торговли вообще, чтобы минимизировать transaction costs. С текущими `transaction_fee=0.04%` + `slippage=0.025%` = **~0.065% per trade**, low-liq тикеры могут добавить ещё 0.5-1.0%.[2][1]

**Рекомендация**: После первого обучения проверь распределение сделок по тикерам в validation — если агент избегает >70% тикеров, стоит отфильтровать самые неликвидные (например, оставить top-150 по volume).

### Длительность сессии 30 минут
**Текущая логика**: 30 шагов при 1-минутных барах = 30 минут торговли.[1]

**Анализ**:
- **30 минут достаточно** для интрадей momentum/mean-reversion стратегий на криптовалютах
- Receptive field 150 баров (2.5 часа) захватывает ~5× больше контекста, чем длина сессии — это хорошо
- С текущим TSL (d0=1.8%, dmin=0.5%) позиция может закрыться за 5-10 минут при быстром движении — агент успеет сделать 2-3 сделки за сессию[2]

**Проблема**: С `gamma=0.9995` дисконтирование фактически отключено (0.9995³⁰ ≈ 0.985). Агент одинаково ценит reward на 1-й и 29-й минуте сессии.[1]

**Альтернативы**:
- **15 минут (15 шагов)** — для scalping с быстрым TSL
- **60 минут (60 шагов)** — для swing внутри часа, но потребует gamma ниже (0.99) для temporal discount
- **Оставить 30 минут, но снизить gamma** до 0.995-0.998 для более чёткого временного приоритета

## План последовательной реализации

Работаем по High Priority → Medium Priority, с проверкой после каждого изменения.

---

### **Итерация 1: Reward Function — добавить risk-awareness**

**Цель**: Штрафовать волатильность reward'а, чтобы агент учился smooth equity curve вместо aggressive PnL swings.

**Подход**: Добавить rolling Sortino-like penalty в step reward.

**Diff для `trading_environment.py`:**

```diff
 class TradingEnvironment(gym.Env):
     def __init__(
         self,
         # ... existing params ...
         max_drawdown_threshold: float | None = None,
         max_drawdown_penalty: float = 0.0,
         max_drawdown_penalty_type: str = "absolute",
+        reward_volatility_penalty: float = 0.0,  # Новый параметр
+        reward_rolling_window: int = 10,          # Окно для расчёта волатильности
         **kwargs,
     ) -> None:
         # ... existing code ...
         self.max_drawdown_threshold = max_drawdown_threshold
         self.max_drawdown_penalty = max_drawdown_penalty
         self.max_drawdown_penalty_type = max_drawdown_penalty_type
+        self.reward_volatility_penalty = reward_volatility_penalty
+        self.reward_rolling_window = reward_rolling_window
```

```diff
 def _init_episode_vars(self) -> None:
     # ... existing code ...
     # Отслеживание просадки
     self.equity_peak: float = self.initial_balance
     self.current_max_drawdown: float = 0.0
+    
+    # Отслеживание волатильности reward
+    self.recent_rewards: List[float] = []  # Буфер последних N rewards
```

```diff
 def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
     # ... existing code до расчёта reward ...
     
     reward = pnl_change / self.initial_balance - inaction_penalty
     
+    # Добавить reward в буфер для расчёта волатильности
+    self.recent_rewards.append(pnl_change / self.initial_balance)
+    if len(self.recent_rewards) > self.reward_rolling_window:
+        self.recent_rewards.pop(0)
+    
+    # Рассчитать волатильность reward (downside deviation для Sortino-like penalty)
+    volatility_penalty = 0.0
+    if len(self.recent_rewards) >= 3 and self.reward_volatility_penalty > 0:
+        rewards_array = np.array(self.recent_rewards)
+        mean_reward = np.mean(rewards_array)
+        # Downside deviation: только отрицательные отклонения от среднего
+        downside_returns = rewards_array[rewards_array < mean_reward] - mean_reward
+        if len(downside_returns) > 0:
+            downside_std = np.sqrt(np.mean(downside_returns ** 2))
+            volatility_penalty = downside_std * self.reward_volatility_penalty
     
     # Calculate current portfolio value (balance + unrealized pnl)
     portfolio_value = self.balance
     # ... existing code ...
     
     # Применить штраф за просадку к награде
     reward -= drawdown_penalty
+    reward -= volatility_penalty
     
     if self.render_mode == "human":
         self._render_human(info, action, reward)
     
     return obs, reward, terminated, False, info
```

**Diff для `config.py`:**

```diff
 class MarketConfig(BaseModel):
     initial_balance: float = 10_000.0
     transaction_fee: float = 0.0004
     slippage: float = 0.0005 / 2
     num_actions: int = 4
     inaction_penalty_ratio: float = 0.001
     bankruptcy_threshold: float = 0.0
     bankruptcy_penalty: float = 1.0
     # Max Drawdown Penalty
     max_drawdown_threshold: float = -0.20
     max_drawdown_penalty: float = 0.1
     max_drawdown_penalty_type: str = "proportional"
+    # Reward Volatility Penalty
+    reward_volatility_penalty: float = 0.0
+    reward_rolling_window: int = 10
```

**Diff для `alpha_seed_404.py`:**

```diff
 cfg.market.max_drawdown_threshold = -0.20
 cfg.market.max_drawdown_penalty_type = "proportional"
 cfg.market.max_drawdown_penalty = 0.1
+
+# Reward Volatility Penalty (Sortino-like)
+cfg.market.reward_volatility_penalty = 0.15  # Начни с 0.1-0.2
+cfg.market.reward_rolling_window = 10        # Окно 10 шагов (~10 минут)
```

**Diff для `train.py`:**

```diff
     envkwargs = {
         # ... existing params ...
         "max_drawdown_threshold": cfg.market.max_drawdown_threshold,
         "max_drawdown_penalty": cfg.market.max_drawdown_penalty,
         "max_drawdown_penalty_type": cfg.market.max_drawdown_penalty_type,
+        "reward_volatility_penalty": cfg.market.reward_volatility_penalty,
+        "reward_rolling_window": cfg.market.reward_rolling_window,
     }
```

**Что это даёт**:
- Агент будет пенализироваться за **резкие проседания** reward (даже если итоговый PnL положительный)
- Motivation для smooth, consistent trades вместо "all-in on lucky momentum"
- Прямая связь с Sortino ratio — минимизируем downside volatility

**Проверка после применения**: Запусти 1-2 эпохи (5-10k steps), проверь что:
1. `reward` в логах не стал слишком негативным (если все rewards < -0.1, уменьши penalty до 0.05)
2. Validation Sortino **не упал** по сравнению с baseline (если упал >20%, это overpenalizing)

***

### **Итерация 2: Gamma Discount — синхронизация с horizon'ом**

**Цель**: Сделать temporal credit assignment более явным.

**Diff для `alpha_seed_404.py`:**

```diff
-cfg.rl.gamma = 0.9995  # Discount
+cfg.rl.gamma = 0.995   # Discount (более сильный для 30-шаговых сессий)
```

**Обоснование**:
- `gamma=0.995` даёт дисконт 0.995³⁰ ≈ **0.86** для reward на 30-м шаге
- Агент будет **предпочитать ранние прибыли** vs отложенные (стимул закрывать profitable trades быстрее)
- Half-life ~138 шагов = ~2.3 сессии (разумно для short-term trading)

**Альтернатива** (если хочешь более агрессивный discount):
```python
cfg.rl.gamma = 0.99  # Half-life ~69 шагов = ~1.15 сессии
```

**Проверка**: Validation WinRate должен **немного вырасти** (агент быстрее фиксирует профит), но PnL per trade может уменьшиться (меньше hold time).

***

### **Итерация 3: Inaction Penalty — снизить aggressive overtrading**

**Цель**: Текущие 0.1% penalty за hold **слишком агрессивны** — агент может overtrade только чтобы избежать penalty.

**Diff для `alpha_seed_404.py`:**

```diff
-cfg.market.inaction_penalty_ratio = 0.001  # 0.1% per hold step
+cfg.market.inaction_penalty_ratio = 0.0002  # 0.02% per hold step
```

**Обоснование**:
- С transaction_fee=0.04% + slippage=0.025% = **0.065% round-trip cost**
- Penalty 0.1% за hold означает "лучше торговать, даже если не уверен", что ведёт к overtrading
- **0.02% penalty** ≈ 1/3 от transaction cost — более сбалансированно

**Альтернатива** (можно попробовать после первого теста):
```python
cfg.market.inaction_penalty_ratio = 0.0  # Вообще убрать penalty
```

**Проверка**: Смотри `episode_closed_trades` в validation — должно быть **1-3 сделки за 30-минутную сессию** (не 5-10).

***

### **Итерация 4: Validation Warmup — раннее sanity check**

**Цель**: Детектировать проблемы (NaN loss, bankruptcy spiral) раньше.

**Diff для `alpha_seed_404.py`:**

```diff
-cfg.trainlog.validation_warmup_steps = 150000  # 25% от бюджета
+cfg.trainlog.validation_warmup_steps = 40000   # ~6.7% от бюджета
```

**Обоснование**:
- Первая validation на 40k steps (~1 эпоха с 4 envs) даст early signal
- Если модель "сходит с ума" (WinRate<0.2, Bankruptcy>50%), узнаешь на ранней стадии
- Оригинальная логика "25% warmup" подходит для стабильных setups, но с мультитикерным датасетом риски выше

**Проверка**: Первая validation должна показать хотя бы **WinRate ~0.3-0.4** и Sortino >-0.5 (даже если модель слабая).

***

### **Итерация 5: Validation Gate — adaptive или fallback criterion**

**Цель**: Не терять best models на ранних этапах обучения из-за строгого gate.

**Diff для `alpha_seed_404.py`:**

```diff
 cfg.validation_gate = {
     "min_sharpe": 0.001,
+    "min_sortino": 0.002,  # Уже есть, хорошо
     "min_profit_factor": 1.10,
-    "max_drawdown_atmost": -1.10,  # Это опечатка? Должно быть -0.10 (10% DD limit)?
+    "max_drawdown_atmost": -0.15,  # 15% MaxDD лимит для сохранения модели
     "min_win_rate": 0.34,
     "min_trades": 600,
     "deny_inf_pf": True,
     "deny_zero_drawdown": True,
+    "save_if_better_than_prev": True,  # NEW: fallback — сохранять если лучше предыдущей best, даже если gate fails
 }
```

**Реализация в `train.py` (если `save_if_better_than_prev` ещё не реализован):**

Нужно проверить текущую логику validation gate. Если строка `save_if_better_than_prev` не работает, придётся добавить логику вручную.

**Проверка**: Лог должен показывать сохранение моделей на ранних этапах (даже если они не проходят полный gate).

***

### **Итерация 6 (опционально): Soft Target Updates**

**Цель**: Stabilize Q-learning с низким LR.

**Diff для `alpha_seed_404.py`:**

```diff
 cfg.rl.lr = 2e-5  # AdamW
-cfg.rl.target_update_freq = 5000  # Hard update каждые 5k steps
+cfg.rl.target_update_freq = 1     # Каждый шаг (для soft update)
+cfg.rl.target_update_tau = 0.005  # Soft update коэффициент (θ_target = τ*θ + (1-τ)*θ_target)
```

**Реализация в `agent.py` (нужно проверить, поддерживается ли soft update):**

Если в `D3QNPERAgent` есть только hard update, нужно добавить:

```python
# В методе update_model после градиента:
if self.soft_update:
    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

**Проверка**: TD loss должен быть **более стабильным** (меньше spikes) в TensorBoard/logs.

***

## Последовательность тестирования

1. **Baseline**: Запусти текущую конфигурацию **без изменений**, сохрани метрики (Sortino, Sharpe, MaxDD, WinRate)
2. **Iteration 1**: Применить **только reward volatility penalty**, запустить 1 полное обучение (15k episodes), сравнить с baseline
3. **Iteration 2**: Добавить **gamma=0.995**, повторить обучение
4. **Iteration 3-4**: Добавить **inaction penalty reduction + validation warmup**, проверить стабильность
5. **Iteration 5-6**: Adaptive gate + soft updates (если нужно)

## Что смотреть в результатах

**Primary metrics** (по приоритету):
1. **Validation Sortino ≥ 0.5** (минимально приемлемо для live)
2. **Validation MaxDD ≤ -0.15** (15% просадка максимум)
3. **Profit Factor ≥ 1.4** (устойчивая edge)

**Secondary metrics**:
4. Win Rate ~0.45-0.55 (не должен быть >0.65 — overfit risk)
5. Closed trades per session ~1-3 (не 10+, признак overtrading)
6. TD loss convergence (stable после 100k steps)

Готов дать diff для **Iteration 1 (reward volatility)** прямо сейчас? Или хочешь сначала запустить baseline без изменений для reference?[2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/21325805/c81a1c54-46ab-4452-8150-7bc8e1cf4d4e/alpha_seed_404.py)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/21325805/f4059f6c-1a1c-4041-ad0e-cd001b33f32d/trading_environment.py)