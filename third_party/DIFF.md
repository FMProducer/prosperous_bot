---

## TL;DR

Ваша интеграция векторизации и конфигурации **в целом корректна**:

* В `configs/alpha.py` есть управляемые флаги производительности и блок `cfg.vec` с `num_envs=2`. 
* В `train.py` корректно подключён `DummyVecEnv`, учтены **реальные переходы** (`transitions_count`) и возвращается средний win-rate из батч-эпизода. 

Однако я вижу **два логических изъяна** в векторизованном цикле (оба типичны для VecEnv):

1. **Неверный `next_state` при `done=True`.**
   В Vectorized-окружениях **происходит авто-reset**, и наблюдение, возвращаемое при `done[i] == True`, — это **первое наблюдение следующего эпизода**, а финальное наблюдение лежит в `info` (стандарт Gym/Gymnasium: `final_observation`/аналогичный ключ) — подробности в доках SB3/Gymnasium. ([stable-baselines3.readthedocs.io][1])
   В вашем `DummyVecEnv` финальное наблюдение кладётся в `info["terminal_observation"]`, затем идёт `reset` и в батч попадает уже **reset-obs**. Сейчас вы записываете в буфер `next_obs_b[i]` и флаг `done`, что склеивает эпизоды. Это надо исправить на использование `terminal_observation`. 

2. **Переучёт наград после завершения отдельных под-эпизодов.**
   Цикл идёт `while not done_mask.all()`, но вы продолжаете суммировать `ep_reward` **для всех** подсред даже после того, как часть уже завершилась (они же авто-resetятся и шагают дальше). Для корректного среднего награды «по одному эпизоду на подсреду» добавляйте награду **только** тем подсредам, которые ещё не считались завершёнными на предыдущем шаге. 

Дополнительно (не критично, но лучше): вызывать `agent.learn()` **каждый шаг** и накапливать `ep_losses` (сейчас — ровно один вызов/эпизод в VecEnv-ветке). Это приближает динамику к одиночному режиму и обычно даёт более стабильные кривые. 

---

## Мини-патч (исправления в `train.py`)

Ниже — **минимальный** unified-diff с точными правками. Он:

* берёт финальное наблюдение из `infos[i]["terminal_observation"]` при `done=True` (см. стандарты авторесета: SB3/Gymnasium). ([stable-baselines3.readthedocs.io][1])
* прекращает накапливать награду для подсред, уже помеченных завершёнными;
* (опционально) возвращает частоту `agent.learn()` «как в одиночном режиме» — по одному вызову на внутренний шаг роллаута.

```diff
*** Begin Patch
*** Update File: train.py
@@
 def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent):
@@
-    while not done_mask.all():
+    while not done_mask.all():
+        prev_done = done_mask.copy()
         actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
         next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)
-        # в DQN/пер меры используем done (без разгадки truncated), как и было в одиночной логике
+        # в DQN/пер меры используем done (без разгадки truncated), как и было в одиночной логике
         for i in range(train_env.num_envs):
-            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_obs_b[i], bool(dones[i]))
+            # Корректный next_state при done: брать финальное наблюдение из info
+            if bool(dones[i]) and isinstance(infos[i], dict):
+                next_state = infos[i].get("terminal_observation", next_obs_b[i])
+            else:
+                next_state = next_obs_b[i]
+            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_state, bool(dones[i]))
             if bool(dones[i]) and isinstance(infos[i], dict):
                 wr = infos[i].get("episode_win_rate", None)
                 if wr is not None:
                     win_rates.append(float(wr))
-        ep_reward += rewards
+        # Накапливать награды только для тех подсред, которые ещё не были завершены до этого шага
+        for i in range(train_env.num_envs):
+            if not prev_done[i]:
+                ep_reward[i] += float(rewards[i])
         obs_batch = next_obs_b
         done_mask |= dones  # эпизод для каждой под-среды
         step_iters += 1
+        # (Опционально) вызывать шаг обучения на каждом батч-шаге, как в одиночной ветке:
+        # loss = agent.learn()
+        # if loss is not None:
+        #     ep_losses.append(loss)
@@
-    avg_reward = float(ep_reward.mean())
+    avg_reward = float(ep_reward.mean())
     avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
     transitions_count = int(step_iters * train_env.num_envs)
     return avg_reward, avg_win_rate, transitions_count
*** End Patch
```

> Обоснование: векторизованные среды **авто-ресетятся**, поэтому наблюдение при `done[i]` — уже **следующий эпизод**, а финальное наблюдение лежит в `info` (Gymnasium: `final_observation`; у вас — `terminal_observation`). Использование финального наблюдения для терминального перехода — рекомендованная практика, прямо отмеченная в руководствах SB3/Gymnasium. ([stable-baselines3.readthedocs.io][1])

---

## Риски/эффекты

| Шаг                                | Что меняется                                           | KPI/риск                                           |
| ---------------------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| Испр. терминального next_state     | Точность данных в буфере опытов (нет склейки эпизодов) | ↑ устойчивость обучения; риск 0                    |
| Стоп-аккумуляция наград после done | Корректные «средние по эпизоду» в VecEnv               | Метрика reward по эпизоду не «перекручена»; риск 0 |
| (Опц.) `learn()` на каждом шаге    | Ближе к одиночной динамике; более частые апдейты       | Времени/степов/с GPU ↑ умеренно                    |

---