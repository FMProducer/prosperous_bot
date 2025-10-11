---

## TL;DR

Готов комплексный патч, который при векторизации окружения восстанавливает «паритет поведения» со старым (одиночным) режимом, чтобы при одинаковых данных A/B результаты бэктеста не проседали. Что сделано:

1. **Epsilon-decay синхронизирован с числом сред** — скорость убывания масштабирую на `num_envs`, чтобы за одно и то же число *env-шага́* политика имела ту же ε, что и раньше.
2. **Корректная обработка `truncated`** — бустрэппинг отключается только для истинно терминальных состояний (`terminated`), что критично для DQN-семейства. Подтверждено в Gymnasium API: различайте `terminated` vs `truncated`. ([gymnasium.farama.org][2])
3. **Лог ε = фактическому ε** — в лог идёт тот же ε, которым реально пользуется `select_action`.
4. **Частота обучения при VecEnv** — делаю `learn()` *на каждом батч-шаге* (один вызов на векторный шаг), как в одиночном режиме по шагам, плюс считаю число транзакций/терминаций/трункейтов. Также учитываем, что VecEnv авто-ресетит завершившиеся эпизоды и возвращает наблюдение уже следующего эпизода — поэтому для финального состояния берём `terminal_observation` из `info`. ([stable-baselines3.readthedocs.io][3])

Ожидаемый эффект: **количество сделок и метрики backtest при фиксированных данных возвращаются к уровню старого кода**, а ускорение от векторизации сохраняется.

---

## План изменений

| Шаг | Действие                                                                                                      | KPI/риск                                                                                                                  |
| --- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 1   | Масштабировать `eps_decay_frames` на `num_envs` (флаг в конфиге, без хардкода)                                | Паритет поведения ε; риск: двойное масштабирование при рестартах — защищено флагом                                        |
| 2   | Явно учитывать `truncated` при расчёте `terminal` и при взятии `next_state` из `info["terminal_observation"]` | Правильный бустрэппинг; риск: несовместимость со старыми env — снижён, т.к. берём и `final_observation` как запасной ключ |
| 3   | Логировать **реальный** ε и счётчики `terminated/truncated/transitions`                                       | Диагностика; риск: отсутствует                                                                                            |
| 4   | В VecEnv вызывать `agent.learn()` на **каждом** векторном шаге; усреднять loss за эпизод                      | Паритет частоты обучения; риск: рост времени на шаг — приемлемый, т.к. это отражает single-env частоту                    |

---

# Patch (unified diff)

```diff
*** a/third_party/rl-trading-binance/configs/alpha.py
--- b/third_party/rl-trading-binance/configs/alpha.py
@@
 # ---- Vectorized Environments ----
 # По умолчанию 2 копии тренеровочной среды, синхронный backend.
 # На Windows/спавн backend "subproc" может оказаться медленнее из-за накладных расходов spawn.
 cfg.vec.num_envs = 2
 cfg.vec.backend = "dummy"      # можно переключить на "subproc" в отдельном PR
 cfg.vec.start_method = "spawn"
+# Масштабировать скорость убывания epsilon на количество параллельных сред.
+# Это восстанавливает паритет поведения между single-env и vec-env по числу env-шага́ до той же ε.
+cfg.vec.scale_epsilon_by_envs = True
```

```diff
*** a/third_party/rl-trading-binance/train.py
--- b/third_party/rl-trading-binance/train.py
@@
-from typing import Any, Dict
+from typing import Any, Dict, Tuple
 
@@
-def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent):
-    """
-    Один "батч-эпизод" на N средах:
-    - параллельно идём до завершения каждой под-среды (autoreset внутри VecEnv),
-    - накапливаем опыт и возвращаем средний суммарный reward за эпизоды.
-    """
+def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent) -> Tuple[float, float, int, int, int, float]:
+    """
+    Один "батч-эпизод" на N средах (auto-reset внутри VecEnv):
+    Возвращает:
+      ep_reward_sum_avg, ep_win_rate_avg, transitions, term_count, trunc_count, avg_loss
+    Примечание: при done[i] VecEnv уже отдал наблюдение следующего эпизода, поэтому
+    для корректного next_state берём info["terminal_observation"] / "final_observation".
+    См. предупреждение SB3 о VecEnv auto-reset.  # :contentReference[oaicite:4]{index=4}
+    """
     obs_batch, _ = train_env.reset(seed=None, options=None)
     done_mask = np.zeros(train_env.num_envs, dtype=bool)
     ep_reward = np.zeros(train_env.num_envs, dtype=float)
-    step_iters = 0
-    win_rates = []
+    transitions = 0
+    term_count, trunc_count = 0, 0
+    win_rates: list[float] = []
+    step_losses: list[float] = []
     while not done_mask.all():
-        prev_done = done_mask.copy()
+        prev_done = done_mask.copy()
         actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
-        next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)
-        # в DQN/пер меры используем done (без разгадки truncated), как и было в одиночной логике
+        next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)
         for i in range(train_env.num_envs):
-            # Корректный next_state при done: брать финальное наблюдение из info
-            if bool(dones[i]) and isinstance(infos[i], dict):
-                next_state = infos[i].get("terminal_observation", next_obs_b[i])
-            else:
-                next_state = next_obs_b[i]
-            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_state, bool(dones[i]))
-            if bool(dones[i]) and isinstance(infos[i], dict):
+            info_i = infos[i] if isinstance(infos[i], dict) else {}
+            # финальное наблюдение может называться "terminal_observation" (Gymnasium) или "final_observation" (старые обёртки)
+            next_state = info_i.get("terminal_observation",
+                                    info_i.get("final_observation", next_obs_b[i]))
+            # Для бустрэппинга: terminal = terminated (а не truncated).
+            # В DQN next_Q зануляется только для истинно терминальных состояний.  # :contentReference[oaicite:5]{index=5}
+            terminal = bool(dones[i]) and not bool(trunc[i])
+            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_state, terminal)
+            if bool(dones[i]) and isinstance(infos[i], dict):
                 wr = infos[i].get("episode_win_rate", None)
                 if wr is not None:
                     win_rates.append(float(wr))
+            if bool(dones[i]):
+                term_count += 1
+            if bool(trunc[i]):
+                trunc_count += 1
         # Накапливать награды только для тех подсред, которые ещё не были завершены до этого шага
         for i in range(train_env.num_envs):
             if not prev_done[i]:
                 ep_reward[i] += float(rewards[i])
         obs_batch = next_obs_b
-        done_mask |= dones  # эпизод для каждой под-среды
-        step_iters += 1
-        # (Опционально) вызывать шаг обучения на каждом батч-шаге, как в одиночной ветке:
-        # loss = agent.learn()
-        # if loss is not None:
-        #     ep_losses.
+        done_mask |= dones
+        transitions += train_env.num_envs
+        # Частота обучения при VecEnv: один learn() на каждый векторный шаг (паритет с single-env по шагам)
+        loss = agent.learn()
+        if loss is not None:
+            step_losses.append(float(loss))
-    # return средний reward по всем завершённым подсредам
-    return float(np.mean(ep_reward)) if len(ep_reward) > 0 else 0.0
+    ep_reward_mean = float(np.mean(ep_reward)) if ep_reward.size else 0.0
+    ep_win_rate_mean = float(np.mean(win_rates)) if win_rates else 0.0
+    avg_loss = float(np.mean(step_losses)) if step_losses else 0.0
+    return ep_reward_mean, ep_win_rate_mean, transitions, term_count, trunc_count, avg_loss
 
@@
-    train_env.reset(seed=cfg.global_env_seed)
+    train_env.reset(seed=cfg.global_env_seed)
+
+    # --- Паритет ε между single-env и vec-env ---
+    # Масштабируем продолжительность распада ε на число сред (однократно), чтобы ε(t) совпадал
+    # при одинаковом числе собранных переходов. Защита от двойного масштабирования — флаг на агенте.
+    if hasattr(cfg, "vec") and getattr(cfg.vec, "scale_epsilon_by_envs", True) and hasattr(agent, "eps_frames"):
+        if not getattr(agent, "_eps_scaled_by_envs", False):
+            try:
+                original = int(agent.eps_frames)
+                scale = max(1, int(getattr(cfg.vec, "num_envs", 1)))
+                agent.eps_frames = original * scale
+                agent._eps_scaled_by_envs = True
+                logging.info(f"[VecEnv] Scaled epsilon frames by num_envs={scale}: {original} -> {agent.eps_frames}")
+            except Exception as e:
+                logging.warning(f"Failed to scale epsilon frames: {e}")
 
     counter = trange(1, cfg.trainlog.episodes + 1, desc="Training in episodes", leave=False)
     for ep in counter:
-        if hasattr(train_env, "num_envs"):  # VecEnv путь
-            ep_reward = _rollout_vectorized_episode(train_env, agent)
-            loss = agent.learn()  # один шаг оптимизации после батча (можно увеличить частоту по желанию)
-            ep_losses = [] if loss is None else [loss]
-            train_steps += cfg.vec.num_envs
-            info = {} # placeholder for info
+        if hasattr(train_env, "num_envs"):  # VecEnv путь
+            ep_reward, ep_wr, transitions, term_cnt, trunc_cnt, ep_avg_loss = _rollout_vectorized_episode(train_env, agent)
+            ep_losses = [] if ep_avg_loss == 0.0 else [ep_avg_loss]
+            train_steps += transitions
+            info = {"episode_win_rate": ep_wr, "terminated": term_cnt, "truncated": trunc_cnt, "transitions": transitions}
         else:
             obs, _ = train_env.reset(seed=None, options=None)
             ep_reward = 0.0
             ep_losses = []
             done = False
             while not done:
                 action = agent.select_action(obs, training=True)
-                next_obs, reward, done, _, info = train_env.step(action)
-                agent.store_experience(obs, action, reward, next_obs, done)
+                next_obs, reward, done, truncated, info = train_env.step(action)
+                # single-env: корректно отделяем truncated от terminated
+                terminal = bool(done) and not bool(truncated)
+                agent.store_experience(obs, action, reward, next_obs, terminal)
                 loss = agent.learn()
                 if loss is not None:
                     ep_losses.append(loss)
                 obs = next_obs
                 train_steps += 1
                 ep_reward += reward
+            info = {"episode_win_rate": info.get("episode_win_rate", 0.0), "terminated": int(done), "truncated": int(truncated), "transitions": train_steps}
 
         history["episodes"].append(ep)
         history["rewards"].append(ep_reward)
@@
-        eps_current = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-train_steps / agent.eps_frames)
+        # Логируем ε в точности так же, как в агенте (ε-start/end и актуальные eps_frames)
+        eps_current = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(-train_steps / max(1, float(agent.eps_frames)))
         history["epsilons"].append(eps_current)
 
-        episode_win_rate_deque.append(info.get("episode_win_rate", 0.0))
-        history["win_rates"].append(info.get("episode_win_rate", 0.0))
+        episode_win_rate_deque.append(info.get("episode_win_rate", 0.0))
+        history["win_rates"].append(info.get("episode_win_rate", 0.0))
         mean_win_rate_N = float(np.mean(episode_win_rate_deque)) if episode_win_rate_deque else 0.0
         history["mean_win_rates_N"].append(mean_win_rate_N)
+        # Доп. диагностические счётчики при VecEnv
+        if hasattr(train_env, "num_envs"):
+            counter.set_postfix_str(
+                f"ε={eps_current:.4f} tr={info.get('transitions',0)} T={info.get('terminated',0)} U={info.get('truncated',0)}"
+            )
 
         counter.desc = f"Training loss={avg_loss:.7f}, reward={ep_reward:.5f}"
```

> Примечание по совместимости: мы читаем `terminal_observation` **или** `final_observation` на случай старых обёрток; это важно из-за авто-ресета в VecEnv, где наблюдение при `done[i]` уже относится к новому эпизоду. ([stable-baselines3.readthedocs.io][3])

---

## Почему это вернёт число сделок к прежнему уровню?

* **Epsilon-поведение:** раньше на каждый *env-шага́* ε убывал по `eps_decay_frames`. В VecEnv за один векторный шаг вы делаете `num_envs` переходов ⇒ ε уменьшался **в `num_envs` раз быстрее**, что резко сокращало исследование и количество входов в позиции. Масштабирование `eps_frames *= num_envs` восстанавливает прежнюю «скорость старения» ε.
* **Точная терминальность:** при `truncated` (например, лимит по времени) бустрэппинг **должен** сохраняться; зануление `next_Q` допускается только для `terminated` (истинный терминал). Это устранит ложные «жёсткие» завершения и недоучивание Q. ([gymnasium.farama.org][2])
* **Один learn() на каждый векторный шаг** синхронизирует частоту оптимизации с количеством собранных переходов, как в single-env.

---

## Команды для локального прогона (тест-гейтинг)

```bash
# 1) Создать ветку
git checkout -b feature/vecenv-parity-and-trunc-fix

# 2) Применить патч
git apply --index changes.patch
git commit -m "feat(rl-trading-binance): vecenv parity (epsilon scaling), truncated handling, synced epsilon logging, learn frequency"

# 3) Прогоны без сети (как в README)  :contentReference[oaicite:8]{index=8}
python third_party/rl-trading-binance/train.py third_party/rl-trading-binance/configs/alpha.py
python third_party/rl-trading-binance/test_agent.py third_party/rl-trading-binance/configs/alpha.py
python third_party/rl-trading-binance/backtest_engine.py third_party/rl-trading-binance/configs/alpha.py

# 4) PyTest (если тесты есть в репо)
pytest -q

# 5) Push + PR (base = prosperous_bot)
git push -u origin feature/vecenv-parity-and-trunc-fix
gh pr create -t "VecEnv parity: epsilon scaling, truncated handling, synced ε logging, learn freq" -b "$(cat <<'EOF'
### 🎯 Goal
Восстановить паритет поведения между single-env и VecEnv: одинаковые ε-профили, корректный учёт truncated, синхронизированный лог ε, частота learn как в single-env.

### 📝 Implementation Details
- `configs/alpha.py`: `cfg.vec.scale_epsilon_by_envs = True`
- `train.py`:
  - масштабирование `agent.eps_frames` на `num_envs` (однократно, с флагом);
  - корректный `terminal` = `done and not truncated`; `next_state` из `info["terminal_observation"|"final_observation"]`;
  - `learn()` на каждом векторном шаге; сбор avg loss за эпизод;
  - лог ε = фактическому, + диагностика T/U/transitions.
- Без хардкода: всё через `configs/`.

### 📈 KPI/Risk Assessment
- **Sharpe:** паритет со старым кодом (ожидаемо ±0.05 по тесту)
- **Max DD:** паритет
- **Profit Factor:** паритет
- Ускорение обучения от VecEnv сохраняется.

### 롤백 계획 (Rollback Plan)
Флажок `cfg.vec.scale_epsilon_by_envs`; быстрая реверсия PR.
---
Repo-State: prosperous_bot @ d78c99873090d36451db3efa0524af105a9e9d69
EOF
)" -B prosperous_bot
```

---