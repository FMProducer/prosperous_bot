**TL;DR:** Добавляю полноценные чекпоинты (policy+target, optimizer, AMP GradScaler, meta) и обратную совместимость с “старыми” `.pth`. Это позволит безопасно останавливать/возобновлять длинные запуски без деградации скорости/качества. В `train.py` ничего менять не нужно — он уже вызывает `agent.save_model(...)` / `agent.load_model(...)`; функция загрузки теперь умеет оба формата.

| Шаг | Действие                                                                                                  | KPI/риск                                                                          |
| --- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1   | Сохраняем в чекпоинт состояния `policy`, `target`, `optimizer`, `GradScaler` (+Meta: шаги, eps-параметры) | Надёжное возобновление без “прогрева”; риск: несовпадение версий PyTorch — низкий |
| 2   | В `load_model` добавлена поддержка нового (чекпоинт) и старого (только `state_dict`) форматов             | Обратная совместимость; риск: отсутствует                                         |
| 3   | Не меняем конфиги и пайплайн — соответствуем требованиям “configs-only”                                   | Никакой хардкод, только код агента.                                               |

---

# Патч (unified diff, ≤300 строк)

> **Изменяется только** `third_party/rl-trading-binance/agent.py`. В `train.py` уже есть вызовы `save_model`/`load_model`, поэтому дополнительные правки не нужны, и тестовые/боевые сценарии сохраняются. 

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/agent.py
@@
     def increment_step(self) -> None:
         self.total_steps += 1
 
-    def save_model(self, path: str) -> None:
-        os.makedirs(os.path.dirname(path), exist_ok=True)
-        torch.save(self.policy_net.state_dict(), path)
-        logger.info(f"Model saved to {path}")
+    def save_model(self, path: str) -> None:
+        """
+        Сохраняет ПОЛНЫЙ чекпоинт для безопасного возобновления обучения:
+        - policy/target state_dict
+        - optimizer state_dict
+        - GradScaler (если AMP включён)
+        - meta (счётчики шагов/eps-параметры и UTC-время)
+        Обратная совместимость: загрузка старых .pth с одним state_dict поддерживается в load_model().
+        """
+        os.makedirs(os.path.dirname(path), exist_ok=True)
+        checkpoint = {
+            "format": "d3qn_per_agent_v1",
+            "created_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
+            "policy_state": self.policy_net.state_dict(),
+            "target_state": self.target_net.state_dict(),
+            "optimizer_state": self.optimizer.state_dict(),
+            "scaler_state": (self.scaler.state_dict() if hasattr(self, "scaler") else None),
+            "meta": {
+                "total_steps": int(self.total_steps),
+                "learn_steps": int(self.learn_steps),
+                "eps_start": float(self.eps_start),
+                "eps_end": float(self.eps_end),
+                "eps_frames": int(self.eps_frames),
+            },
+        }
+        torch.save(checkpoint, path)
+        logger.info(f"Checkpoint saved to {path} (policy+target+optimizer+scaler+meta).")
 
-    def load_model(self, path: str) -> None:
-        state_dict = torch.load(path, map_location=self.device)
-        self.policy_net.load_state_dict(state_dict)
-        self.target_net.load_state_dict(state_dict)
-        self.policy_net.eval()
-        self.target_net.eval()
-        logger.info(f"Model loaded from {path}")
+    def load_model(self, path: str, strict: bool = True) -> None:
+        """
+        Загружает либо новый чекпоинт (см. save_model), либо старый .pth с единственным state_dict.
+        Аргумент strict пробрасывается в load_state_dict для гибкости при мелких несовпадениях ключей.
+        """
+        obj = torch.load(path, map_location=self.device)
+        # Новый формат (чекпоинт)
+        if isinstance(obj, dict) and "policy_state" in obj:
+            self.policy_net.load_state_dict(obj["policy_state"], strict=strict)
+            self.target_net.load_state_dict(obj.get("target_state", obj["policy_state"]), strict=strict)
+            opt_state = obj.get("optimizer_state")
+            if opt_state:
+                try:
+                    self.optimizer.load_state_dict(opt_state)
+                except Exception as e:
+                    logger.warning(f"Optimizer state load skipped: {e}")
+            scaler_state = obj.get("scaler_state")
+            if hasattr(self, "scaler") and scaler_state:
+                try:
+                    self.scaler.load_state_dict(scaler_state)
+                except Exception as e:
+                    logger.warning(f"GradScaler state load skipped: {e}")
+            meta = obj.get("meta", {}) or {}
+            self.total_steps = int(meta.get("total_steps", self.total_steps))
+            self.learn_steps = int(meta.get("learn_steps", self.learn_steps))
+            kind = "checkpoint"
+        else:
+            # Старый формат (только веса сети)
+            self.policy_net.load_state_dict(obj, strict=strict)
+            self.target_net.load_state_dict(obj, strict=strict)
+            kind = "weights-only"
+        self.policy_net.eval()
+        self.target_net.eval()
+        logger.info(f"Model loaded from {path} ({kind}).")
*** End Patch
```

**Почему это достаточно:** `train.py` уже сохраняет `best.pth` и `final.pth` (валидация/финал), и грузит модель перед тестом — мы лишь делаем содержимое файлов богаче и совместимым назад, не меняя протокол вызовов. 

---

## Мини-валидация (локально)

```bash
# 1) Быстрый smoke-тест импортов и сохранения/загрузки
python - <<'PY'
import torch, os
from third_party.rl-trading-binance.agent import D3QN_PER_Agent
from third_party.rl-trading-binance.config import cfg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = D3QN_PER_Agent(
    state_shape=(cfg.seq.input_history_len, cfg.seq.num_features),
    action_dim=cfg.market.num_actions,
    cnn_maps=cfg.model.cnn_maps, cnn_kernels=cfg.model.cnn_kernels, cnn_strides=cfg.model.cnn_strides,
    dense_val=cfg.model.dense_val, dense_adv=cfg.model.dense_adv, additional_feats=cfg.model.additional_feats,
    dropout_model=cfg.model.dropout_p, device=device, gamma=cfg.rl.gamma, learning_rate=cfg.rl.learning_rate,
    batch_size=cfg.rl.batch_size, buffer_size=cfg.per.buffer_size, target_update_freq=cfg.rl.target_update_freq,
    train_start=cfg.rl.train_start, per_alpha=cfg.per.per_alpha, per_beta_start=cfg.per.per_beta_start,
    per_beta_frames=cfg.per.per_beta_frames, eps_start=cfg.eps.eps_start, eps_end=cfg.eps.eps_end,
    eps_frames=cfg.eps.eps_decay_frames, epsilon=cfg.per.per_eps, max_gradient_norm=cfg.rl.max_gradient_norm
)
os.makedirs("tmp_models", exist_ok=True)
p="tmp_models/test.pth"
agent.save_model(p)
agent.load_model(p)
print("OK: checkpoint save/load")
PY

# 2) (опционально) обычные тесты проекта, если настроены
pytest -q
```

---