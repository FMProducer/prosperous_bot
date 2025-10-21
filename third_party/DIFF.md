## TL;DR

Логи `Starting index generation...` и `--> Processing symbol: ...` означают, что **идёт построение индекса окон из БД**: берётся диапазон ~3 месяцев и **23 тикера** из `configs/alpha.py` — это тяжёлая операция по минутным данным, поэтому всё выглядит «долго», но это корректное поведение. После завершения будет создан `output/alpha/stream_backtest_index.csv`, далее запустится исполнение сделок и сохранение `trades.csv`/`metrics.json`. (См. логи и код ветки построения индекса в `paper_trader.py`.)  

При этом я заметил одну важную вещь по инференсу:

* Ваша политика может вернуть **"HOLD"** (см. `inference_adapter.py`). Сейчас `_policy_to_side(...)` в `paper_trader.py` интерпретирует любые нечисловые ответы как SELL — это приведёт к **ложным продажам**, когда модель хотела «не входить». Рекомендую мини-патч: корректно обрабатывать {0, "HOLD"} → пропуск окна.  

Дополнительно — микро-улучшение логирования: печатать число найденных окон на тикер и итоговый счётчик — чтобы вы видели прогресс не только по именам.

---

## Почему индексация «долго» — и это норма

* В `alpha.py` включён режим `build_index_from_db=True`, задан большой период `2025-03-01 → 2025-06-01` и список из **23 символов**. Для каждого символа PaperTrader запрашивает минутки и minute-by-minute сканирует всплески по алгоритму 90→10 (look-ahead=True). Это вычислительно и I/O-интенсивно. Логи `Starting index generation...` / `--> Processing symbol: ...` печатаются как раз в этой ветке.  
* После окончания цикла собирается `DataFrame` окон и **сразу приводится к UTC-datetime** (важно для `.dt.floor("D")` ниже), затем пишется CSV. Это в коде уже сделано правильно. 

Если нужно «быстрый дымовой тест» — сократите `data.symbols` до 1–2 тикеров и/или сузьте `time_range` до пары дней; а затем верните прод-параметры.

---

## Мини-патч: корректный "HOLD" + информативный прогресс

Ниже — компактный unified diff к вашему текущему `paper_trader.py`. Он:

1. Учит `_policy_to_side(...)` правильно обрабатывать **HOLD/0** (пропуск окна при `strict=True`),
2. Добавляет счётчик и печать количества окон на каждый тикер и итоговый total.

```diff
*** a/third_party/rl-trading-binance/paper_trader.py
--- b/third_party/rl-trading-binance/paper_trader.py
@@
-def _policy_to_side(policy: _Policy, symbol: str, df_ctx: pd.DataFrame) -> Optional[str]:
+def _policy_to_side(policy: _Policy, symbol: str, df_ctx: pd.DataFrame) -> Optional[str]:
     # Универсальный вызов с мягкой деградацией интерфейса
     if hasattr(policy, "predict_side"):
-        side = policy.predict_side(df_ctx)  # ожидается "BUY"/"SELL"
-        return str(side).upper()
+        side = str(policy.predict_side(df_ctx)).upper()
+        if side in ("BUY", "SELL"):
+            return side
+        if side == "HOLD":
+            return None
+        return None
     if hasattr(policy, "predict"):
-        pred = policy.predict(symbol, df_ctx)
-        try:
-            pred = int(pred)
-        except Exception:
-            pass
-        return "BUY" if pred == 1 else "SELL"
+        pred = policy.predict(symbol, df_ctx)
+        # допускаем как str, так и int
+        if isinstance(pred, str):
+            up = pred.upper()
+            if up in ("BUY", "SELL"):
+                return up
+            if up == "HOLD":
+                return None
+            return None
+        try:
+            pred_i = int(pred)
+        except Exception:
+            return None
+        if pred_i == 1:
+            return "BUY"
+        if pred_i == 2:
+            return "SELL"
+        # 0 или иное — трактуем как HOLD/нет сигнала
+        return None
     if callable(policy):
         pred = policy(df_ctx)
         try:
             pred = int(pred)
         except Exception:
             pass
-        return "BUY" if pred == 1 else "SELL"
+        if pred == 1:
+            return "BUY"
+        if pred == 2:
+            return "SELL"
+        return None
     return None
@@
-    if need_build:
+    if need_build:
         if cfg.time_start_utc is None or cfg.time_end_utc is None or not cfg.symbols:
             raise RuntimeError(
                 "Для построения индекса из БД нужны data.time_range{start_utc,end_utc} и data.symbols[]. "
                 "Либо выключите build_index_from_db=False и подготовьте stream_backtest_index.csv офлайн."
             )
         rows = []
         print("Starting index generation...")
+        total_wins = 0
         for sym in cfg.symbols:
             print(f"--> Processing symbol: {sym}")
             feed = dict(provider([sym], cfg.time_start_utc.isoformat(), cfg.time_end_utc.isoformat()))
             if sym not in feed or feed[sym].empty:
                 continue
             df = _ensure_utc_index(feed[sym]).sort_index()
@@
-            for (ctx_start, ctx_end, ses_start, ses_end, abs_chg) in wins:
+            for (ctx_start, ctx_end, ses_start, ses_end, abs_chg) in wins:
                 # Приводим торговую сессию к длине из конфига (напр., 10 минут в демо)
                 ses_end_adj = ses_start + timedelta(minutes=cfg.session_minutes)
                 rows.append({
                     "symbol": sym,
                     "ctx_start": ctx_start.isoformat(),
                     "ctx_end": ctx_end.isoformat(),
                     "session_start": ses_start.isoformat(),
                     "session_end": ses_end_adj.isoformat(),
                     "abs_change_pct": abs_chg,
                 })
+            print(f"    -> windows found: {len(wins)}")
+            total_wins += len(wins)
         idx = pd.DataFrame(rows)
         out_dir = os.path.dirname(cfg.index_csv)
         os.makedirs(out_dir, exist_ok=True)
         # Сразу храним UTC-датавремена и используем их же ниже
         for col in ["ctx_start","ctx_end","session_start","session_end"]:
             idx[col] = pd.to_datetime(idx[col], utc=True)
         idx.to_csv(cfg.index_csv, index=False)
+        print(f"Index saved: {cfg.index_csv}  | total windows: {total_wins}")
@@
-        side = _policy_to_side(policy, sym, df_ctx)
+        side = _policy_to_side(policy, sym, df_ctx)
 
-        if side not in ("BUY", "SELL"):
+        if side not in ("BUY", "SELL"):
             if cfg.inference.strict:
                 continue
             else:
                 raise RuntimeError(f"predict_side вернул некорректное значение: {side}")
```

**Эффект:** при ответе модели `"HOLD"`/`0` — окно пропускается (в `strict=True` это именно то, что нужно), BUY/SELL — исполняем. Плюс вы теперь видите прогресс по количеству найденных окон на каждый тикер и итоговый total.

---

## Быстрые рекомендации по производительности (без кода)

| Шаг | Что сделать                                                                                                                                                                                                | Зачем                                                             |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1   | Для проверки пайплайна временно сузьте `data.time_range` (например, на 2–3 дня) и оставьте 1–2 символа в `data.symbols`.                                                                                   | Быстрый end-to-end прогон, убедиться в корректности окон/трейдов. |
| 2   | Если нужен лимит на количество окон уже при построении индекса, а не только после — можно добавить «ранний колпак» per-symbol (остановка при достижении N). (Сейчас колпак применяется после чтения CSV.)  | Сократить время I/O/CPU при большом периоде.                      |
| 3   | Убедитесь, что `db_provider:get_feed` действительно отдаёт **все ожидаемые каналы** (цены, объёмы). Если чего-то нет — `inference_adapter` сам дополняет колонки нулями, это уже реализовано.              | Исключить рассинхронизацию формы входа.                           |

---

## Контрольные пункты по инференсу (важно)

* Нарезка окна для модели: вы уже делаете строго `master_cfg.seq.agent_history_len` минут перед `ctx_end` — это правильно и устраняет прошлую ошибку `Window shape mismatch in apply_normalization`. 
* Политика может возвращать BUY/SELL/HOLD; теперь HOLD корректно пропускается. 
* В `alpha.py` `inference.strict=True`: так вы получаете **1:1 соответствие с бэктестом** по действиям (без эвристик). 

---

## Что делать прямо сейчас

1. Примените мини-патч выше к `paper_trader.py` (он не меняет интерфейсов и укладывается в лимиты PR).
2. Для «быстрого успеха» временно уменьшите `data.symbols`/`time_range`, прогоните, убедитесь, что `stream_backtest_index.csv` появился и дальше отработали `trades.csv`/`metrics.json`.
3. Верните прод-настройки и запускайте полноценную индексацию.