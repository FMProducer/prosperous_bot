---

## TL;DR

**fallback к Follow-Context** в `paper_trader.py` сейчас только путает — для реальной симуляции нам нужен **жёсткий режим**: *модель не загрузилась → немедленная ошибка, никакой подмены поведения*. Ниже — маленький патч: добавляю в конфиг `data.inference.strict=True` и меняю `paper_trader.py`, чтобы при strict-режиме **запрещался любой fallback** (только указанная модель). Путь к чекпойнту берётся из конфига. Конфиги — строго в `configs/*.py`, артефакты — в `output/<config_name>/`.

---

## Почему раньше был fallback и почему отключаем

* Исторически Follow-Context служил «дым-тестом» для контура исполнения, когда модели ещё нет. Это отражено в текущем описании скрипта: baseline Follow-Context. 
* В вашем процессе пейпер-трейдинга нужна **конкретная обученная модель** из конфига. Значит, правильная политика — **fail-fast**: нет модели/ошибка загрузки → падение с понятной диагностикой, а не скрытая замена логики. Это соответствует регламенту о конфиг-драйве и воспроизводимости. 

---

## Патч (unified diff, строго по файлам проекта)

### 1) Конфиг: включаем строгий инференс и задаём чекпойнт по умолчанию

```diff
*** a/third_party/rl-trading-binance/configs/alpha.py
--- b/third_party/rl-trading-binance/configs/alpha.py
@@
     # Если файл провайдера лежит рядом (db_provider.py), используем прямой импорт:
     "db_provider": "db_provider:get_feed",
+    # ---- Inference (строгий режим без фоллбэка) ----
+    "inference": {
+        "policy_loader": "inference_adapter:load_policy",  # module:function
+        "checkpoint_path": r"C:\Python\Prosperous_Bot\third_party\rl-trading-binance\output\fmproducer_1_eval\saved_models\session_1\best.pth",
+        "strict": True
+    },
     # ---- Paper trading (RT/ASAP) ----
     "paper_trader": {
         "mode": "asap",             # "realtime" | "asap"
         "cap_windows_per_symbol": 0 # 0 = без лимита; иначе макс. окон/день/тикер
     },
```

*(Конфиги — только в `configs/*.py`.)*

### 2) Трейдер: если `strict=True` и модель не загрузилась — **raise**, без fallback

```diff
*** a/third_party/rl-trading-binance/paper_trader.py
--- b/third_party/rl-trading-binance/paper_trader.py
@@
-from tqdm import tqdm
+from tqdm import tqdm
 import heapq
-from typing import Protocol, Any
+from typing import Protocol, Any
 
@@
 class PTParams:
     mode: str  # "realtime" | "asap"
     cap_windows_per_symbol: int
 
+@dataclass
+class InferenceParams:
+    policy_loader: str
+    checkpoint_path: str
+    strict: bool
+
 @dataclass
 class Cfg:
     config_name: str
     db_provider_path: str
     index_csv: str
     exec: ExecParams
     pt: PTParams
+    inf: InferenceParams
 
 def _load_cfg(cfg_path: str) -> Cfg:
@@
-    pt_d = data.get("paper_trader", {"mode": "asap", "cap_windows_per_symbol": 0})
-    ex_d = data.get("exec", {})
+    pt_d = data.get("paper_trader", {"mode": "asap", "cap_windows_per_symbol": 0})
+    ex_d = data.get("exec", {})
+    inf_d = data.get("inference", {})
@@
     ptp = PTParams(
         mode=str(pt_d.get("mode", "asap")),
         cap_windows_per_symbol=int(pt_d.get("cap_windows_per_symbol", 0)),
     )
-    return Cfg(config_name, dbp, index_csv, execp, ptp)
+    infp = InferenceParams(
+        policy_loader=str(inf_d.get("policy_loader", "")),
+        checkpoint_path=str(inf_d.get("checkpoint_path", "")),
+        strict=bool(inf_d.get("strict", False)),
+    )
+    return Cfg(config_name, dbp, index_csv, execp, ptp, infp)
@@
-# ------------------------------ Strategy / Policy -------------------
-# Плагинная политика инференса; fallback — Follow-Context.
+# ------------------------------ Strategy / Policy -------------------
+# Плагинная политика инференса; при strict=True — без fallback.
 
 class Policy(Protocol):
     def predict(self, symbol: str, ctx_df: pd.DataFrame) -> str:
         """Вернуть 'BUY' или 'SELL' по данным контекста."""
         ...
 
-def _load_policy(loader_path: str, ckpt_path: str) -> Policy | None:
+def _load_policy(loader_path: str, ckpt_path: str) -> Policy | None:
     try:
         if ":" not in loader_path:
             raise RuntimeError("`inference.policy_loader` должен быть 'module:function'.")
         mod_path, fn_name = loader_path.split(":", 1)
         mod = importlib.import_module(mod_path)
         if not hasattr(mod, fn_name):
             raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}`.")
         loader = getattr(mod, fn_name)
         policy = loader(ckpt_path)  # type: ignore
         return policy
     except Exception as e:
-        print(f"[paper_trader] WARN: не удалось загрузить политику из '{loader_path}' ({e}). Использую Follow-Context.")
+        print(f"[paper_trader] WARN: не удалось загрузить политику из '{loader_path}' ({e}).")
         return None
@@
-    provider = _load_db_provider(cfg.db_provider_path)
-    policy = _load_policy(cfg.inf.policy_loader, cfg.inf.checkpoint_path)
+    provider = _load_db_provider(cfg.db_provider_path)
+    policy = _load_policy(cfg.inf.policy_loader, cfg.inf.checkpoint_path) if cfg.inf.policy_loader else None
+    if cfg.inf.strict and policy is None:
+        raise RuntimeError("[paper_trader] strict=True: модель не загружена — прерываю симуляцию.")
@@
-        # Выбор стороны:
-        # 1) Если доступна модель — спрашиваем политику на контексте [ctx_start, ctx_end].
-        # 2) Иначе fallback: Follow-Context по знаку ретёрна.
-        side = None
+        # Выбор стороны:
+        # 1) Если доступна модель — спрашиваем политику на контексте [ctx_start, ctx_end].
+        # 2) Если strict=False и модель недоступна — fallback (Follow-Context).
+        side = None
         ctx_slice = df.loc[ctx_start:ctx_end]
         if policy is not None:
             try:
                 side = str(policy.predict(sym, ctx_slice))
                 if side not in ("BUY","SELL"):
                     raise ValueError("policy returned non-standard action")
             except Exception as e:
-                print(f"[paper_trader] WARN: policy.predict() failed ({e}); fallback to Follow-Context.")
-                side = None
-        if side is None:
+                if cfg.inf.strict:
+                    raise
+                print(f"[paper_trader] WARN: policy.predict() failed ({e}); fallback → Follow-Context.")
+                side = None
+        if side is None:  # fallback используется ТОЛЬКО при strict=False
             px_ctx_start = float(ctx_slice.iloc[0]["close"])
             px_ctx_end   = float(ctx_slice.iloc[-1]["close"])
             ctx_ret = (px_ctx_end / max(px_ctx_start, 1e-12)) - 1.0
             if ctx_ret == 0.0:
                 continue
             side = "BUY" if ctx_ret > 0.0 else "SELL"
```

> В текущем `paper_trader.py` действительно есть упоминание baseline Follow-Context и конфиг без inference-секции — патч добавляет именно строгий инференс по модели из конфига.

---

## Шаги запуска и гейты

| Шаг | Действие                                                                     | KPI/риск                                                                      |
| --- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 1   | Применить патч и убедиться, что путь к `.pth` в `configs/alpha.py` корректен | Конфиг читается; путь существует                                              |
| 2   | Запуск трейдера                                                              | При `strict=True` симуляция стартует **только** с успешно загруженной моделью |
| 3   | Быстрая проверка `trades.csv`: есть и BUY, и SELL; нет скрытого LONG-bias    | PF/Sharpe реалистичнее                                                        |
| 4   | При любой ошибке загрузки/инференса — процесс падает с явным сообщением      | Нулевая двусмысленность, как вы просили                                       |

---

## Почему это соответствует правилам проекта

* Все параметры — **в `configs/*.py`**, без хардкодов. 
* Артефакты остаются в `output/<config_name>/`. 
* Изменения точечные, с минимальным контекстом и в пределах лимитов PR. 

Если хотите полностью исключить даже код fallback-ветки, могу в следующем патче **удалить** её целиком — тогда `paper_trader.py` будет компилироваться и работать **только** при наличии и корректности модели.
