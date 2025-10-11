---

## TL;DR

## Мелкие несоответствия (не критично)

1. **`tqdm(total)` в `evaluate_agent`:** сейчас `range(1, num_episodes + 1)` и `total=num_episodes + 1` — прогресс-бар будет «длиннее» на 1. Предлагаю `total=num_episodes`. 

2. **Совпадение ключей метрик c графиками теста:** `plot_test_distributions()` ожидает `Test_all_reward` и `Test_all_win_rate`, а `evaluate_agent()` возвращает только `..._all_pnls`. Из-за этого видите предупреждения и пустые графики (код так и пишет warning). Предлагаю добавить два списка в `metrics`. 

---

## Мини-патч (unified diff ≤ 30 строк)

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/train.py
@@
-    for ep in tqdm(range(1, num_episodes + 1), total=num_episodes + 1, desc=f"{split_name} in episodes", leave=False):
+    for ep in tqdm(range(1, num_episodes + 1), total=num_episodes, desc=f"{split_name} in episodes", leave=False):
@@
-    metrics = {
-        f"{split_name}_mean_reward": np.mean(rewards),
-        f"{split_name}_mean_pnl": np.mean(pnls),
-        f"{split_name}_win_rate": np.mean(win_rates),
-        f"{split_name}_all_pnls": pnls,
-    }
+    metrics = {
+        f"{split_name}_mean_reward": float(np.mean(rewards)),
+        f"{split_name}_mean_pnl": float(np.mean(pnls)),
+        f"{split_name}_win_rate": float(np.mean(win_rates)),
+        f"{split_name}_all_pnls": pnls,
+        # добавить ключи, которых ждут тестовые графики:
+        f"{split_name}_all_reward": rewards,
+        f"{split_name}_all_win_rate": win_rates,
+    }
*** End Patch
```

**Почему это безопасно:** не меняет тренировочную логику; только косметика прогресс-бара и совместимость ключей метрик с уже написанной функцией построения графиков.

---

## Шаг | Действие | KPI/риск

1 | Оставить реализацию `terminal_observation/final_observation` и `done & ~truncated` как есть | Корректные таргеты DQN; ↓ ложных терминалов. Риск 0. ([The Farama Foundation][2])
2 | Применить мини-патч на tqdm/метрики | Чистые логи и полноценные тест-графики. Риск 0. 
3 | (Опционально) масштабировать частоты `val_freq/save_freq` при `num_envs>1` (делить на `num_envs`) | Сопоставимость частот с single-env. Риск 0. ([stable-baselines3.readthedocs.io][1])

---

### Справки (почему «так правильно»)

* **VecEnv авто-reset и «финальный кадр в info»**: при `done[i]` наблюдение — уже старт нового эпизода; последний кадр хранится как `final_observation`/`terminal_observation`. ([gymnasium.farama.org][4])
* **`terminated` vs `truncated`**: `truncated` — внешний лимит (напр., по времени), и это **не** терминал — таргеты надо бутстрэпить. ([The Farama Foundation][2])
* **Экспоненциальный ε-decay**: стандартная формула и практика логирования из фактического ε. ([docs.pytorch.org][3])