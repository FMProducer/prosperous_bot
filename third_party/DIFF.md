---

### Мини-патч (точечные фиксы `agent.py`)

Ниже единый diff (≤20 строк) для двух моментов: (A) аккуратное переключение train/eval в `get_mean_std_q()`, (B) безопасная перезапись кэша (атомарная запись).

```diff
*** a/third_party/rl-trading-binance/agent.py
--- b/third_party/rl-trading-binance/agent.py
@@
     def get_mean_std_q(self, state: np.ndarray, n_samples: int = 5) -> Tuple[float, float]:
-        self.policy_net.train()  # включаем стохастику (dropout)
+        # Включаем стохастику (dropout), но сохраняем и восстанавливаем исходный режим
+        prev_training = self.policy_net.training
+        self.policy_net.train()
         x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
         q_list = []
         with torch.no_grad():
             for _ in range(n_samples):
                 q = self.policy_net(x).squeeze(0).detach().cpu().numpy()
                 q_list.append(q)
-        q_arr = np.stack(q_list, axis=0)
+        # Восстанавливаем исходный режим (детерминированный инференс вне MC-оценки)
+        if not prev_training:
+            self.policy_net.eval()
+        q_arr = np.stack(q_list, axis=0)
         return float(q_arr.mean()), float(q_arr.std(ddof=1) if n_samples > 1 else 0.0)
@@
     def save_disk_cache(self) -> None:
-        if not os.path.exists(self.cache_path):
-            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
-            with open(self.cache_path, "wb") as f:
-                pickle.dump(self.qval_cache, f)
-            logger.info(f"Q-value cache saved at {self.cache_path}")
+        # Атомарная перезапись кэша: временный файл + os.replace
+        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
+        tmp_path = self.cache_path + ".tmp"
+        with open(tmp_path, "wb") as f:
+            pickle.dump(self.qval_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
+            f.flush(); os.fsync(f.fileno())
+        os.replace(tmp_path, self.cache_path)
+        logger.info(f"Q-value cache saved at {self.cache_path}")
```

Обоснование:
— Возврат режима `eval()` предотвращает случайно включённый dropout в инференсе (стабильность/воспроизводимость). ([PyTorch Forums][5])
— Атомарная запись защищает от повреждения файла при сбоях/долгих прогонах. ([Python in Plain English][6])

---

## Краткий план и риски

| Шаг | Действие                                                  | KPI/риск                                             |
| --- | --------------------------------------------------------- | ---------------------------------------------------- |
| 1   | Исправить `get_mean_std_q()` (восстанавливать режим)      | +стабильность инференса; ↓дрейф метрик на валидации  |
| 2   | Сделать атомарной `save_disk_cache()`                     | +надёжность долгих прогонов; ↓риск коррупции кэша    |
| 3   | (Опционально) Вызов `policy_net.eval()` в путях инференса | +детерминизм действий вне тренировки                 |
| 4   | (Опционально) `non_blocking=True` при `.to(device)`       | +1–3% к скорости при `pin_memory=True` в DataLoader  |

---

## Соответствие конфиг-политике

Все переключатели производительности (AMP, `torch.compile`, CuDNN) у нас действительно управляются через `configs/*.py` (`PerformanceConfig`), как и требуют правила проекта — без хардкода. 

---