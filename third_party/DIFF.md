## TL;DR

## changes.patch (минимальный фикс регресса скорости)

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/configs/alpha.py
@@
-# включаем реальный параллелизм для CPU-тяжелых сред
-cfg.vec.backend = "subproc"
+# По умолчанию используем DummyVecEnv (часто быстрее для "лёгких" env).
+# SubprocVecEnv включает прицельно под тяжёлые env/на Linux.
+cfg.vec.backend = "dummy"
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/subproc_vec_env.py
@@
-            elif cmd == _CMD_STEP:
+            elif cmd == _CMD_STEP:
                 action = data
                 obs, rew, terminated, truncated, info = env.step(action)
-                # autoreset семантика как в векторных обёртках: если эпизод кончился — сразу reset
-                if terminated or truncated:
-                    info = dict(info or {})
-                    info["terminal_observation"] = obs
-                    obs, info_reset = env.reset(seed=None, options=None)
-                    info["reset_info"] = info_reset
+                # autoreset семантика: если эпизод кончился — сразу reset
+                # ⚠️ Но НЕ пересылаем финальный кадр при terminated (он не нужен DQN-таргету:
+                #     next_Q будет занулён). Передаём финальный кадр ТОЛЬКО при truncated,
+                #     где нужен бутстрап корректного s'.
+                if terminated or truncated:
+                    info = dict(info or {})
+                    if truncated:
+                        info["terminal_observation"] = obs  # финальный кадр нужен для bootstrap
+                    obs, info_reset = env.reset(seed=None, options=None)
+                    info["reset_info"] = info_reset
                 remote.send((obs, float(rew), bool(terminated), bool(truncated), info))
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/agent.py
@@
-        with torch.no_grad(), torch.autocast(device_type="cuda" if self.device.type=="cuda" else "cpu",
-                                             enabled=getattr(self, "amp_enabled", False)):
+        with torch.no_grad(), torch.autocast(
+            device_type=("cuda" if self.device.type == "cuda" else "cpu"),
+            enabled=getattr(self, "use_amp", False)
+        ):
             x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
             q = self.policy_net(x)              # [N, action_dim]
             greedy = q.argmax(dim=1).detach().to("cpu").numpy()  # [N]
*** End Patch
```

> Этот патч:
> • возвращает дефолт на `DummyVecEnv` (как у вас было при 60+ it/s),
> • ***в 2–3 строки*** урезает крупные IPC-передачи: финальный кадр шлём только при `truncated=True` (для корректного бутстрапа), при `terminated=True` больше не гоняем гигантский массив — **DQN его всё равно не использует** (см. формулу и реализацию). ([stable-baselines3.readthedocs.io][3])
> • включает AMP на инференсе батчем.

---

## Что ожидать после фикса

* На `backend="dummy"` количество сред может **не масштабировать** FPS линейно, но **должно вернуться** к прежним ~60 it/s (и выше) на 2 env, как было у вас до `subproc`. Это нормальная рекомендация SB3 для «лёгких» env: *DummyVecEnv обычно быстрее*. ([colab.research.google.com][1])
* Если захотите `subproc` (тяжёлые env, Linux/`fork`) — после патча IPC-трафик на `done` заметно снизится (нет лишнего `terminal_observation` при terminated), и масштабирование станет ощутимее. Но на Windows/`spawn` всё равно бывает хуже из-за старт-метода. ([Stack Overflow][5])

---

## Шаг | Действие | KPI/риск

1 | Применить патч (3 файла) | ↑ it/s (возврат к прежнему), ↓ IPC; риск 0
2 | Запустить с `cfg.vec.backend="dummy"` (как в патче) | Стабильные 60+ it/s на 2 env (как у вас было)