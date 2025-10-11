## TL;DR

1. В счётчиках вы увеличиваете `term_count` при **любом** `done`, даже если это `truncated=True`. Для чистой статистики лучше считать **только истинные терминалы**.  ([The Farama Foundation][2])
2. Логируемый ε вы рассчитываете формулой по `train_steps`. Это ок, но **ещё надёжнее** брать его из самого агента (атрибут `epsilon` или `get_epsilon()`), чтобы график ε точно совпадал с тем, что реально использует политика. 

Ниже минимальный патч на оба пункта.

---

## changes.patch (микро-фиксы статистики и логирования ε)

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/train.py
@@
-            if bool(dones[i]):
-                term_count += 1
+            # считать "terminated" только когда done и НЕ truncated
+            if bool(dones[i]) and not bool(trunc[i]):
+                term_count += 1
             if bool(trunc[i]):
                 trunc_count += 1
@@
-        # Логируем ε в точности так же, как в агенте (ε-start/end и актуальные eps_frames)
-        eps_current = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-train_steps / max(1, float(agent.eps_frames)))
-        history["epsilons"].append(eps_current)
+        # Логируем ФАКТИЧЕСКИЙ ε агента (если доступен), иначе падать назад на формулу
+        eps_current = None
+        if hasattr(agent, "epsilon"):
+            try:
+                eps_current = float(agent.epsilon)
+            except Exception:
+                eps_current = None
+        if eps_current is None and hasattr(agent, "get_epsilon") and callable(agent.get_epsilon):
+            try:
+                eps_current = float(agent.get_epsilon())
+            except Exception:
+                eps_current = None
+        if eps_current is None:
+            eps_current = float(
+                agent.eps_end + (agent.eps_start - agent.eps_end)
+                * np.exp(-train_steps / max(1.0, float(agent.eps_frames)))
+            )
+        history["epsilons"].append(eps_current)
*** End Patch
```

### Почему так

* **VecEnv & финальные наблюдения.** При авто-ресете наблюдение при `done[i]` — уже **первый кадр нового эпизода**, а финальный кадр лежит в `info["terminal_observation"]/["final_observation"]`. Ваш код это учитывает; патч не меняет эту часть. ([stable-baselines3.readthedocs.io][1]) 
* **`truncated` ≠ `terminated`.** По API Gymnasium, тайм-лимит (`truncated=True`) **не** терминал — таргеты DQN нужно бутстрэпить. Исправление `term_count` делает статистику честной. ([The Farama Foundation][2])
* **Логирование ε.** Считывание ε прямо из агента гарантирует полное совпадение с политикой `select_action`; при отсутствии атрибута используем формулу-fallback (как сейчас). 

---

## Быстрый чек-лист после мержа

| Шаг | Действие                                                                              | KPI/риск                                         |
| --- | ------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 1   | Сравнить `transitions/terminated/truncated` в логах на одинаковом периоде             | Диагностика поведения эпизодов; риск 0           |
| 2   | Проверить график ε: совпадает с внутренним ε агента                                   | Честные A/B; риск 0                              |
| 3   | Пересмотреть частоты `val_freq/save_freq` при `n_envs>1` (обычно делят на `num_envs`) | Стабильность логов/сейвов; риск 0. ([GitHub][3]) |

---