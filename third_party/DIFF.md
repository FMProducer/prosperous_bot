---

## TL;DR

Готов патч, который добавляет **настраиваемое количество копий сред** через конфиг (по умолчанию `2`) и включает **векторизованный сбор опыта** в обучающем цикле. Реализация без новых внешних зависимостей: свой лёгкий `DummyVecEnv` (синхронно в одном процессе). Это безопасная ступень №1. Позже (если будет нужно) можно добавить `SubprocVecEnv` (мультипроцессный backend) тем же интерфейсом. Концептуально это согласуется с Gymnasium/Stable-Baselines3 (`SyncVectorEnv/AsyncVectorEnv`, `DummyVecEnv/SubprocVecEnv`). ([gymnasium.farama.org][2])

---

## changes.patch

> Все пути — внутри `third_party/rl-trading-binance/`. Патч затрагивает **3 файла** (1 новый) и ~220 строк diff. Конфиг-параметры — в отдельном `VecConfig`.

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/config.py
@@
-from typing import List, Literal, Optional
+from typing import List, Literal, Optional
 
 import torch
 from pydantic import BaseModel, Field, validator
@@
 class PathConfig(BaseModel):
@@
     def cache_dir(self) -> str:
         return os.path.join(self.output_dir, "backtest_qval_cache")
 
+class VecConfig(BaseModel):
+    """
+    Параметры векторизации окружений (Vectorized Environments).
+    По умолчанию включаем 2 копии тренеровочной среды и синхронный backend.
+    """
+    num_envs: int = 2
+    backend: Literal["dummy", "subproc"] = "dummy"  # "dummy" = 1 процесс, синхронно
+    start_method: Literal["spawn", "fork", "forkserver"] = "spawn"  # безопасно на всех ОС
+
 class PerformanceConfig(BaseModel):
@@
 class MasterConfig(BaseModel):
@@
-    device: DeviceConfig = DeviceConfig()
+    device: DeviceConfig = DeviceConfig()
     paths: PathConfig = PathConfig()
+    vec: VecConfig = VecConfig()
     perf: PerformanceConfig = PerformanceConfig()
     data: DataConfig = DataConfig()
     seq: SequenceConfig = SequenceConfig()
     market: MarketConfig = MarketConfig()
     rl: RLConfig = RLConfig()
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/configs/alpha.py
@@
 # ---- Performance presets for i5-6600 + GTX 1070 (Pascal) ----
@@
 cfg.perf.compile_mode = "reduce-overhead"
 
+# ---- Vectorized Environments ----
+# По умолчанию 2 копии тренеровочной среды, синхронный backend.
+# На Windows/спавн backend "subproc" может оказаться медленнее из-за накладных расходов spawn.
+cfg.vec.num_envs = 2
+cfg.vec.backend = "dummy"      # можно переключить на "subproc" в отдельном PR
+cfg.vec.start_method = "spawn"
+
 # python train.py configs/alpha.py
 # python test_agent.py configs/alpha.py
 # python backtest_engine.py configs/alpha.py
 # python optimize_cfg.py configs/alpha.py
*** End Patch
```

```diff
*** Begin Patch
*** Add File: third_party/rl-trading-binance/vec_env.py
+from __future__ import annotations
+import numpy as np
+from typing import Callable, List, Tuple, Any, Sequence
+
+class DummyVecEnv:
+    """
+    Минимальная синхронная векторизация без зависимостей (аналог gym Sync/DummyVecEnv):
+    - хранит N независимых копий среды в одном процессе;
+    - reset()/step() работают с батчами наблюдений/действий;
+    - при done/terminated/ truncated — авто-reset соответствующей под-среды.
+    """
+    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
+        assert len(env_fns) >= 1, "Need at least one env_fn"
+        self.envs = [fn() for fn in env_fns]
+        self.num_envs = len(self.envs)
+
+    def reset(self, seed=None, options=None):
+        obs_batch, infos = [], []
+        for i, env in enumerate(self.envs):
+            s = None if seed is None else (seed + i if isinstance(seed, int) else None)
+            obs, info = env.reset(seed=s, options=options)
+            obs_batch.append(obs)
+            infos.append(info)
+        return np.stack(obs_batch), infos
+
+    def step(self, actions: Sequence[Any]):
+        assert len(actions) == self.num_envs, "actions must match num_envs"
+        obs_b, rew_b, done_b, trunc_b, infos = [], [], [], [], []
+        for env, act in zip(self.envs, actions):
+            next_obs, reward, done, truncated, info = env.step(act)
+            # autoreset для закончившихся эпизодов
+            if done or truncated:
+                info = dict(info or {})
+                info["terminal_observation"] = next_obs
+                next_obs, info_reset = env.reset(seed=None, options=None)
+                info["reset_info"] = info_reset
+            obs_b.append(next_obs)
+            rew_b.append(float(reward))
+            done_b.append(bool(done))
+            trunc_b.append(bool(truncated))
+            infos.append(info)
+        return (
+            np.stack(obs_b),
+            np.asarray(rew_b, dtype=float),
+            np.asarray(done_b, dtype=bool),
+            np.asarray(trunc_b, dtype=bool),
+            infos,
+        )
+
+    def close(self):
+        for env in self.envs:
+            close = getattr(env, "close", None)
+            if callable(close):
+                close()
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/train.py
@@
-from agent import D3QN_PER_Agent
-from config import MasterConfig
+from agent import D3QN_PER_Agent
+from config import MasterConfig
+from vec_env import DummyVecEnv
@@
-from trading_environment import TradingEnvironment
+from trading_environment import TradingEnvironment
@@
-def main(cfg: MasterConfig) -> None:
+def _make_train_env_fns(env_kwargs, n: int):
+    # фабрика копий среды для векторизации
+    return [lambda ek=env_kwargs: TradingEnvironment(**ek) for _ in range(n)]
+
+def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent):
+    """
+    Один "батч-эпизод" на N средах:
+    - параллельно идём до завершения каждой под-среды (autoreset внутри VecEnv),
+    - накапливаем опыт и возвращаем средний суммарный reward за эпизоды.
+    """
+    obs_batch, _ = train_env.reset(seed=None, options=None)
+    done_mask = np.zeros(train_env.num_envs, dtype=bool)
+    ep_reward = np.zeros(train_env.num_envs, dtype=float)
+    while not done_mask.all():
+        actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
+        next_obs_b, rewards, dones, trunc, infos = train_env.step(actions)
+        # в DQN/пер меры используем done (без разгадки truncated), как и было в одиночной логике
+        for i in range(train_env.num_envs):
+            agent.store_experience(obs_batch[i], actions[i], float(rewards[i]), next_obs_b[i], bool(dones[i]))
+        ep_reward += rewards
+        obs_batch = next_obs_b
+        done_mask |= dones  # эпизод для каждой под-среды
+    return float(ep_reward.mean())
+
+def main(cfg: MasterConfig) -> None:
@@
-    train_env = TradingEnvironment(**env_kwargs)
+    # --- TRAIN ENV: single vs vectorized ---
+    if cfg.vec.num_envs > 1:
+        train_env = DummyVecEnv(_make_train_env_fns(env_kwargs, cfg.vec.num_envs))
+        logging.info(f"Vectorized train env: DummyVecEnv x{cfg.vec.num_envs}")
+    else:
+        train_env = TradingEnvironment(**env_kwargs)
@@
-    for ep in counter:
-        obs, _ = train_env.reset(seed=None, options=None)
-        ep_reward = 0.0
-        ep_losses = []
-        done = False
-
-        while not done:
-            action = agent.select_action(obs, training=True)
-            next_obs, reward, done, _, info = train_env.step(action)
-            agent.store_experience(obs, action, reward, next_obs, done)
-            loss = agent.learn()
-            if loss is not None:
-                ep_losses.append(loss)
-            obs = next_obs
-            train_steps += 1
+    for ep in counter:
+        if hasattr(train_env, "num_envs"):  # VecEnv путь
+            ep_reward = _rollout_vectorized_episode(train_env, agent)
+            loss = agent.learn()  # один шаг оптимизации после батча (можно увеличить частоту по желанию)
+            ep_losses = [] if loss is None else [loss]
+            train_steps += cfg.vec.num_envs
+        else:
+            obs, _ = train_env.reset(seed=None, options=None)
+            ep_reward = 0.0
+            ep_losses = []
+            done = False
+            while not done:
+                action = agent.select_action(obs, training=True)
+                next_obs, reward, done, _, info = train_env.step(action)
+                agent.store_experience(obs, action, reward, next_obs, done)
+                loss = agent.learn()
+                if loss is not None:
+                    ep_losses.append(loss)
+                obs = next_obs
+                train_steps += 1
@@
-        # logging/plots below (unchanged)
+        # logging/plots ниже (без изменений)
*** End Patch
```

> Примечания к интеграции: мы **векторизуем только train-среду** (валидация/тест — по-старому, на 1 среде), что соответствует распространённой практике: собирать батчи опыта параллельно, а оценку держать детерминированной и сравнимой. Это также упрощает смену частот логирования/сохранений: помните, что один «батч-эпизод» в VecEnv эквивалентен `n_envs` обычным эпизодам по шагам. SB3 прямо предупреждает пересчитать частоты `eval_freq/save_freq` на `// n_envs`. ([stable-baselines3.readthedocs.io][4])

---

## Что это даст и почему так безопаснее

* **Скорость:** теоретически до ~`min(n_envs, ядер CPU/IO узких мест)` ускорения по сбору опыта. В синхронном бэкенде накладные минимальны, и на Windows он часто предпочтительней (нет `spawn` процессов). ([stable-baselines3.readthedocs.io][3])
* **Простота и совместимость:** без новых зависимостей (не тянем Gym/Gymnasium/SB3), API среды не меняем; только точечные ветки в `train.py`.
* **Масштабирование позже:** при желании можно добавить `SubprocVecEnv` с multiprocessing (`spawn`/`forkserver`), но это отдельный небольшой PR; PyTorch/CPython советуют избегать «грязного fork», особенно при многопоточности, поэтому `spawn` остаётся дефолтом. ([Python documentation][5])

---