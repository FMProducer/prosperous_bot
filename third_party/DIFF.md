**Контекст проблемы.** Падение с ~60 it/s до ~8 it/s и отсутствие эффекта от увеличения «сессий» почти наверняка означает, что сейчас вы используете **синхронный однопроцессный** `DummyVecEnv` (векторизация без реального параллелизма) и/или делаете **по одному forward на каждую среду** вместо батч-инференса на GPU. `DummyVecEnv` не задействует несколько ядер, а GIL мешает распараллелить CPU-тяжёлые шаги сред потоками; для таких случаев нужен **multiprocessing** (`SubprocVecEnv`). ([stable-baselines3.readthedocs.io][1])
Дополнительно, если векторизация вызывает **несколько forward подряд** (по одному на каждую копию среды), вы получаете чистые накладные расходы Python и проигрыш — вместо выигрыша. Нужно **батчировать выбор действий**: подавать все N состояний в сеть разом и делать один forward с `torch.no_grad()`, затем разбирать argmax по строкам (классическая DQN-схема). ([docs.pytorch.org][2])
Наконец, **`torch.compile`** в режиме `reduce-overhead` даёт профит только если граф стабилен; при динамике/частых «граф-брейках» он может ухудшить скорость. Держим это как тумблер в конфиге для A/B. ([docs.pytorch.org][3])

---

## TL;DR

Я подготовил патч, который:

1. добавляет **`SubprocVecEnv`** (реальный мультипроцессный бэкенд) и переключатель `cfg.vec.backend = "subproc"`;
2. реализует **батч-выбор действий** в агенте (`select_action_batch`) — один forward на весь батч, без лишних питоновских циклов;
3. в `train.py` автоматически использует `SubprocVecEnv` при `backend="subproc"` и батч-выбор действий, с откатом к поэлементному пути при отсутствии метода.

Это закрывает оба ключевых источника регресса скорости: отсутствие реального параллелизма и отсутствие батча в инференсе.
Док-основания: векторные среды должны **батчить наблюдения/действия** и при тяжёлых шагах выигрывают от **subprocess-бэкенда**, а не от dummy; батч-инференс — каноника DQN. ([gymnasium.farama.org][4])

---

# changes.patch (≤ 300 строк, 4 файла)

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/configs/alpha.py
@@
 # ---- Vectorized Environments ----
 cfg.vec.num_envs = 2
-# по умолчанию ранее стоял "dummy" (однопроцессный, синхронный)
-cfg.vec.backend = "dummy"
+# включаем реальный параллелизм для CPU-тяжелых сред
+cfg.vec.backend = "subproc"
 cfg.vec.start_method = "spawn"
 # Масштабировать скорость убывания epsilon на количество параллельных сред.
 cfg.vec.scale_epsilon_by_envs = True
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/train.py
@@
-from vec_env import DummyVecEnv
+from vec_env import DummyVecEnv
+from subproc_vec_env import SubprocVecEnv
@@
-def _make_train_env_fns(env_kwargs, n: int):
+def _make_train_env_fns(env_kwargs, n: int):
     # фабрика копий среды для векторизации
     return [lambda ek=env_kwargs: TradingEnvironment(**ek) for _ in range(n)]
 
+def _make_vec_env(env_kwargs, cfg):
+    n = int(getattr(cfg.vec, "num_envs", 1))
+    if n <= 1:
+        return TradingEnvironment(**env_kwargs)
+    backend = getattr(cfg.vec, "backend", "dummy")
+    if backend == "subproc":
+        start = getattr(cfg.vec, "start_method", "spawn")
+        return SubprocVecEnv(_make_train_env_fns(env_kwargs, n), start_method=start)
+    # fallback: dummy
+    return DummyVecEnv(_make_train_env_fns(env_kwargs, n))
+
@@
-def _rollout_vectorized_episode(train_env: DummyVecEnv, agent: D3QN_PER_Agent) -> Tuple[float, float, int, int, int, float]:
+def _rollout_vectorized_episode(train_env, agent: D3QN_PER_Agent) -> Tuple[float, float, int, int, int, float]:
@@
-        actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
+        # Батч-выбор действий: один forward на весь набор состояний, если агент это поддерживает
+        if hasattr(agent, "select_action_batch"):
+            actions = agent.select_action_batch(obs_batch, training=True)
+        else:
+            actions = [agent.select_action(obs_batch[i], training=True) for i in range(train_env.num_envs)]
@@
-    # --- TRAIN ENV: single vs vectorized ---
-    if cfg.vec.num_envs > 1:
-        train_env = DummyVecEnv(_make_train_env_fns(env_kwargs, cfg.vec.num_envs))
-        logging.info(f"Vectorized train env: DummyVecEnv x{cfg.vec.num_envs}")
-    else:
-        train_env = TradingEnvironment(**env_kwargs)
+    # --- TRAIN ENV: single vs vectorized ---
+    train_env = _make_vec_env(env_kwargs, cfg)
+    if hasattr(train_env, "num_envs"):
+        logging.info(f"Vectorized train env: backend={getattr(cfg.vec,'backend','dummy')} x{train_env.num_envs}")
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/agent.py
@@
     def select_action(self, state: np.ndarray, training: bool = True) -> int:
         # существующая одиночная версия выбора действия
         ...
+
+    def select_action_batch(self, states: np.ndarray, training: bool = True) -> list[int]:
+        """
+        Векторизованный epsilon-greedy для батча состояний (N,H,W,C/…):
+        - один forward сети на весь батч (torch.no_grad, AMP при включённом autocast)
+        - argmax по действиям для каждой строки
+        - случайные действия под epsilon по тем же индексам
+        Возвращает список длины N.
+        """
+        self.policy_net.eval()  # детерминированный инференс вне MC-дропаут
+        n = int(states.shape[0])
+        with torch.no_grad(), torch.autocast(device_type="cuda" if self.device.type=="cuda" else "cpu",
+                                             enabled=getattr(self, "amp_enabled", False)):
+            x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
+            q = self.policy_net(x)              # [N, action_dim]
+            greedy = q.argmax(dim=1).detach().to("cpu").numpy()  # [N]
+        # epsilon-greedy по батчу
+        eps = float(self.epsilon if hasattr(self, "epsilon") else
+                    (self.eps_end + (self.eps_start - self.eps_end)
+                     * np.exp(-self.total_steps / max(1, self.eps_frames))))
+        if training and eps > 0.0:
+            rnd = np.random.rand(n) < eps
+            if np.any(rnd):
+                rand_actions = np.random.randint(0, self.action_dim, size=int(rnd.sum()))
+                greedy = greedy.copy()
+                greedy[rnd] = rand_actions
+        return greedy.tolist()
*** End Patch
```

```diff
*** Begin Patch
*** Add File: third_party/rl-trading-binance/subproc_vec_env.py
+from __future__ import annotations
+import multiprocessing as mp
+from typing import Callable, Sequence, Any, Tuple, List
+import numpy as np
+
+_CMD_RESET = "reset"
+_CMD_STEP = "step"
+_CMD_CLOSE = "close"
+
+def _worker(remote, env_fn):
+    env = env_fn()
+    try:
+        while True:
+            cmd, data = remote.recv()
+            if cmd == _CMD_RESET:
+                seed, options = data
+                obs, info = env.reset(seed=seed, options=options)
+                remote.send((obs, info))
+            elif cmd == _CMD_STEP:
+                action = data
+                obs, rew, terminated, truncated, info = env.step(action)
+                # autoreset семантика как в векторных обёртках: если эпизод кончился — сразу reset
+                if terminated or truncated:
+                    info = dict(info or {})
+                    info["terminal_observation"] = obs
+                    obs, info_reset = env.reset(seed=None, options=None)
+                    info["reset_info"] = info_reset
+                remote.send((obs, float(rew), bool(terminated), bool(truncated), info))
+            elif cmd == _CMD_CLOSE:
+                remote.close()
+                break
+            else:
+                raise RuntimeError(f"Unknown cmd: {cmd}")
+    except KeyboardInterrupt:
+        pass
+    finally:
+        try:
+            close = getattr(env, "close", None)
+            if callable(close):
+                close()
+        except Exception:
+            pass
+
+class SubprocVecEnv:
+    """
+    Лёгкая мультипроцессная векторизация среды:
+    - каждый воркер держит собственную копию env;
+    - общение через Pipe (минимальный IPC);
+    - интерфейс совместим с DummyVecEnv (reset -> stack, step -> stack).
+    Подходит для CPU-тяжёлых окружений, где DummyVecEnv/потоки упираются в GIL. 
+    """
+    def __init__(self, env_fns: Sequence[Callable[[], Any]], start_method: str = "spawn"):
+        assert len(env_fns) >= 1, "Need at least one env_fn"
+        self.num_envs = len(env_fns)
+        ctx = mp.get_context(start_method)
+        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
+        self.procs: List[mp.Process] = []
+        for work_remote, fn in zip(self.work_remotes, env_fns):
+            p = ctx.Process(target=_worker, args=(work_remote, fn), daemon=True)
+            p.start()
+            work_remote.close()
+            self.procs.append(p)
+
+    def reset(self, seed=None, options=None):
+        for r in self.remotes:
+            r.send((_CMD_RESET, (seed, options)))
+        results = [r.recv() for r in self.remotes]
+        obs, infos = zip(*results)
+        return np.stack(obs), list(infos)
+
+    def step(self, actions: Sequence[Any]):
+        assert len(actions) == self.num_envs, "actions must match num_envs"
+        for r, a in zip(self.remotes, actions):
+            r.send((_CMD_STEP, a))
+        results = [r.recv() for r in self.remotes]
+        obs, rews, terms, truncs, infos = zip(*results)
+        return (np.stack(obs),
+                np.asarray(rews, dtype=float),
+                np.asarray(terms, dtype=bool),
+                np.asarray(truncs, dtype=bool),
+                list(infos))
+
+    def close(self):
+        for r in self.remotes:
+            try:
+                r.send((_CMD_CLOSE, None))
+            except Exception:
+                pass
+        for p in self.procs:
+            try:
+                p.join(timeout=0.2)
+            except Exception:
+                pass
*** End Patch
```

---

## Почему это решит вашу деградацию

* **Реальный параллелизм.** `SubprocVecEnv` обходит GIL, запуская **каждую** среду в отдельном процессе → нагрузка на все ядра CPU, что и требуется для ускорения CPU-тяжёлых сред. Потоки/`DummyVecEnv` такого эффекта не дают. ([realpython.com][5])
* **Батч-инференс.** Один forward по батчу состояний убирает N-кратные накладные вызовов Python и лучше загружает GPU; argmax по строкам — стандарт для DQN. ([docs.pytorch.org][2])
* **Совместимость векторного API.** Мы сохраняем семантику авто-ресета и «финального кадра в `info`» (`terminal_observation`), как предписывает Gym/Gymnasium Vector API и SB3. ([gymnasium.farama.org][4])

---

## Быстрая проверка после мержа

| Шаг               | Что смотреть                                                                | Ожидание                                                     |
| ----------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------ |
| CPU загрузка      | Все 4 ядра i5-6600 должны быть заняты при `backend=subproc`                 | ~80–100% на ядро (во время step)                             |
| Throughput (it/s) | 2 среды → рост it/s vs single-env; 3 — добавит, пока не упрётесь в CPU/диск | > 8 it/s, целимcя вернуться к 60+ (зависит от env-стоимости) |
| GPU util          | Должна **подрасти** за счёт батч-инференса                                  | Скачки при forward, меньше «простоя»                         |
| Логи              | ε и ratio updates/collect должны быть сопоставимы со старым кодом           | стабильные кривые                                            |

---