---

## TL;DR

Проверил ваши версии `paper_trader.py` и `configs/alpha.py`. Сейчас в трейдере всё ещё есть **ветка fallback** (Follow-Context) при `strict=False` и при ошибке `policy.predict(...)` — её нужно **удалить полностью**. Также для инференса модели мы должны грузить данные **с контекстом**, а не только «сессию». Я сделал минимальный патч:

* Полностью исключил fallback-логику (теперь трейдер **работает только при наличии и корректности модели**).
* Гружу OHLCV **за диапазон `[ctx_start, session_end]`**, чтобы передавать в модель реальный контекст.
* Исправил выход из позиции на **последнюю минуту сессии** (`session_end - 1m`), чтобы исключить off-by-one.

Ниже — diff и как применить.

---

## Что не так в текущих файлах

* **`paper_trader.py`** (ваша версия):

  1. В коде есть ветка fallback: при `strict=False` или исключении в `policy.predict()` трейдер уходит в Follow-Context. Это прямо видно в блоке `if side is None: ... side = "BUY" if ctx_ret > 0 else "SELL"`. 
  2. Для инференса срез `ctx_slice = df.loc[ctx_start:ctx_end]` строится на данных, загруженных **только за сессию** (`ses_start→ses_end`) — контекста в DataFrame нет. Это приводит к пустому `ctx_slice` и нестабильному поведению. 
  3. Выход берётся по бару `session_end` (а должен — **по последней минуте сессии**, `session_end - 1m`). 

* **`configs/alpha.py`** (ваша версия): секция `inference` уже есть и стоит `strict: True` и путь к чекпойнту — ок. 

---

## Патч (unified diff)

```diff
*** a/third_party/rl-trading-binance/paper_trader.py
--- b/third_party/rl-trading-binance/paper_trader.py
@@
-Читает индекс эпизодов из stream_backtest_engine → воспроизводит сделки
-в режиме "реального времени" (sleep) или ASAP, подгружая минутки из БД
-только на период сессии. Стратегия по умолчанию: Follow-Context.
+Читает индекс эпизодов из stream_backtest_engine → воспроизводит сделки
+в режиме "реального времени" (sleep) или ASAP, подгружая минутки из БД.
+Работает ТОЛЬКО с указанной в конфиге моделью (без fallback).
@@
-from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Protocol, Any
+from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Protocol, Any
@@
-# Плагинная политика инференса; при strict=True — без fallback.
+# Плагинная политика инференса; fallback ОТСУТСТВУЕТ (модель обязательна).
@@
 def _load_policy(loader_path: str, ckpt_path: str) -> Policy | None:
@@
-    except Exception as e:
-        print(f"[paper_trader] WARN: не удалось загрузить политику из '{loader_path}' ({e}).")
-        return None
+    except Exception as e:
+        raise RuntimeError(f"[paper_trader] Ошибка загрузки политики '{loader_path}': {e}")
@@
-    provider = _load_db_provider(cfg.db_provider_path)
-    policy = _load_policy(cfg.inf.policy_loader, cfg.inf.checkpoint_path) if cfg.inf.policy_loader else None
-    if cfg.inf.strict and policy is None:
-        raise RuntimeError("[paper_trader] strict=True: модель не загружена — прерываю симуляцию.")
+    provider = _load_db_provider(cfg.db_provider_path)
+    policy = _load_policy(cfg.inf.policy_loader, cfg.inf.checkpoint_path)
@@
-    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Paper trading"):
+    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Paper trading"):
         sym = row["symbol"]
         ctx_start = _to_utc(row["ctx_start"])
         ctx_end = _to_utc(row["ctx_end"])
         ses_start = _to_utc(row["session_start"])
         ses_end = _to_utc(row["session_end"])
-        # Подгружаем ровно сессию
-        feed = dict(provider([sym], ses_start.isoformat(), ses_end.isoformat()))
+        # Подгружаем диапазон с КОНТЕКСТОМ и СЕССИЕЙ (для инференса модели):
+        feed = dict(provider([sym], ctx_start.isoformat(), ses_end.isoformat()))
         if sym not in feed or feed[sym].empty:
             continue
         df = _ensure_utc_index(feed[sym]).sort_index()
-        if df.index[0] > ses_start or df.index[-1] < ses_end:
+        # Минимальные проверки покрытия ключевых временных меток
+        last_ts = ses_end - pd.Timedelta(minutes=1)
+        if df.index[0] > ctx_start or df.index[-1] < last_ts:
             # неполное покрытие — пропустим окно
             continue
-        first_px = float(df.loc[ses_start:ses_start].iloc[0]["close"])
-        # Выбор стороны:
-        # 1) Если доступна модель — спрашиваем политику на контексте [ctx_start, ctx_end].
-        # 2) Если strict=False и модель недоступна — fallback (Follow-Context).
-        side = None
-        ctx_slice = df.loc[ctx_start:ctx_end]
-        if policy is not None:
-            try:
-                side = str(policy.predict(sym, ctx_slice))
-                if side not in ("BUY","SELL"):
-                    raise ValueError("policy returned non-standard action")
-            except Exception as e:
-                if cfg.inf.strict:
-                    raise
-                print(f"[paper_trader] WARN: policy.predict() failed ({e}); fallback → Follow-Context.")
-                side = None
-        if side is None:  # fallback используется ТОЛЬКО при strict=False
-            px_ctx_start = float(ctx_slice.iloc[0]["close"])
-            px_ctx_end   = float(ctx_slice.iloc[-1]["close"])
-            ctx_ret = (px_ctx_end / max(px_ctx_start, 1e-12)) - 1.0
-            if ctx_ret == 0.0:
-                continue
-            side = "BUY" if ctx_ret > 0.0 else "SELL"
+        first_px = float(df.loc[ses_start:ses_start].iloc[0]["close"])
+        # Выбор стороны: ТОЛЬКО модель (никаких fallback).
+        ctx_slice = df.loc[ctx_start:ctx_end]
+        try:
+            side = str(policy.predict(sym, ctx_slice))
+        except Exception as e:
+            raise RuntimeError(f"[paper_trader] policy.predict() error for {sym} @ {ctx_end}: {e}")
+        if side not in ("BUY", "SELL"):
+            raise RuntimeError(f"[paper_trader] policy returned invalid action: {side!r}")
@@
-        # Выход в конце сессии
-        last_px = float(df.loc[ses_end:ses_end].iloc[-1]["close"])
+        # Выход по ПОСЛЕДНЕЙ минуте сессии (исключаем бар, начинающийся в session_end)
+        last_px = float(df.loc[last_ts:last_ts].iloc[-1]["close"])
         exit_px = _apply_slippage(last_px, cfg.exec.slippage_bps, "SELL" if side=="BUY" else "BUY")
```

> Ключевые изменения: удалены все ветви кода, где мог запускаться Follow-Context, и добавлены два гейта корректности данных (наличие контекста и точка выхода). Основано на вашей версии файлов. 

---

## Как применить

| Шаг | Действие                                             | KPI/риск                             |
| --- | ---------------------------------------------------- | ------------------------------------ |
| 1   | Сохраните diff выше как `changes.patch` в корне репо | —                                    |
| 2   | Примените и закоммитьте                              | Код собирается                       |
| 3   | Прогоните трейдер                                    | `trades.csv`, `metrics.json` созданы |
| 4   | В логах не должно быть «fallback» сообщений вовсе    | Гарантия «model-only»                |

```bash
git checkout -b feature/paper-trader-no-fallback
git apply --index changes.patch
git commit -m "fix(paper_trader): remove all fallback paths; require model; load [ctx_start, session_end]; exit at session_end-1m"
git push -u origin feature/paper-trader-no-fallback
gh pr create -t "paper_trader: model-only (no fallback), context-aware feed" -B prosperous_bot -b "### 🎯 Goal
Исключить любой fallback; трейдер работает только при наличии корректной модели.

### 📝 Implementation Details
- Полностью удалены ветки Follow-Context.
- Загрузка данных за [ctx_start, session_end] для инференса.
- Выход по последней минуте сессии (session_end-1m).

### 📈 KPI/Risk Assessment
- **Sharpe/PF:** реалистичнее (без скрытой эвристики).
- **Max DD:** без изменений в расчёте, зависит от модели.

### 롤백 계획
Revert PR."
```

---

## Примечания по соответствию регламенту

* Конфиг-драйв: путь к модели и загрузчик — в `configs/*.py` (у вас уже есть `data["inference"]`). 
* Артефакты — в `third_party/rl-trading-binance/output/<config_name>/…`, как требовалось. Текущий `paper_trader.py` сохраняет `trades.csv` и `metrics.json`. 
* Безопасность дат/денег: UTC-времена, суммы в USDT — без изменений. 

Готов, при необходимости, сразу добавить минимальные unit-тесты (pytest) на: (1) падение при отсутствии модели, (2) корректность точки выхода `session_end-1m`, (3) что запрос к провайдеру охватывает `[ctx_start, session_end]` (моки).

[1]: https://github.com/FMProducer/prosperous_bot/commit/6c2e8f94fc9565a2732f32bb47d771b804754b49 "docs: rl-trading-binance-source-code fix · FMProducer/prosperous_bot@6c2e8f9 · GitHub"
