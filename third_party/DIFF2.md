# Repo-State Header (prosperous_bot)

> Не удалось верифицировать удалённый REPO_URL; работаю строго по предоставленным локальным файлам. Целевые пути: `third_party/rl-trading-binance/paper_trader.py`, `third_party/rl-trading-binance/configs/alpha.py`. В текущем `paper_trader.py` индексация строится через `find_spike_windows(...)` (детектор всплесков). 

## TL;DR

Внедряю **режим `index_mode="sliding"`**: окна формируются **на каждом минутном баре** (контекст `ctx_minutes`, сессия `session_minutes`), что соответствует вашему требованию «плавающее 10-минутное окно на каждом новом минутном баре». По умолчанию остаётся прежний режим `spike`. Изменения минимальны: +2 поля конфига, переключатель в месте построения индекса и небольшой построитель окон.

---

## Unified diff (минимальный патч)

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/paper_trader.py
@@
-from dataclasses import dataclass
-from datetime import datetime, timezone, timedelta
-from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Callable, Protocol, Any
+from dataclasses import dataclass
+from datetime import datetime, timezone, timedelta
+from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Callable, Protocol, Any
@@
 class Cfg:
     config_name: str
     db_provider_path: str
     index_csv: str
     exec: ExecParams
     pt: PTParams
     inference: InferenceParams
     # --- расширения для потокового построения индекса ---
     build_index_from_db: bool
     time_start_utc: Optional[datetime]
     time_end_utc: Optional[datetime]
     ctx_minutes: int
     session_minutes: int
+    # режим построения индекса: "spike" | "sliding"
+    index_mode: str
+    sliding_stride_minutes: int
     # детектор
     det_context: int
     det_window: int
     det_abs_change_pct: float
     det_contrast_min: float
     det_cooldown: int
     det_use_lookahead: bool
     # опционально: список тикеров для сканирования
     symbols: List[str]
@@
     ctx_m = int(data.get("ctx_minutes", 30))
     sess_m = int(data.get("session_minutes", 10))
+    index_mode = str(data.get("index_mode", "spike"))
+    sliding_stride = int(data.get("sliding_stride_minutes", 1))
     det = data.get("detector", {})
     det_ctx = int(det.get("context_minutes", 90))
     det_win = int(det.get("window_minutes", 10))
     det_abs = float(det.get("abs_change_pct", data.get("trigger", {}).get("abs_change_pct", 5.0)))
     det_con = float(det.get("contrast_min", 5.0))
     det_cool = int(det.get("cooldown_minutes", data.get("trigger", {}).get("cooldown_minutes", 60)))
     det_la = bool(det.get("use_lookahead", True))
     symbols = list(data.get("symbols", []))
-    paper_trader_cfg = Cfg(config_name, dbp, index_csv, execp, ptp, inf,
-               bool(data.get("build_index_from_db", False)),
-               t_start, t_end, ctx_m, sess_m,
-               det_ctx, det_win, det_abs, det_con, det_cool, det_la,
-               symbols)
+    paper_trader_cfg = Cfg(config_name, dbp, index_csv, execp, ptp, inf,
+               bool(data.get("build_index_from_db", False)),
+               t_start, t_end, ctx_m, sess_m,
+               index_mode, sliding_stride,
+               det_ctx, det_win, det_abs, det_con, det_cool, det_la,
+               symbols)
     return paper_trader_cfg, master_cfg
@@
-            df = _ensure_utc_index(feed[sym]).sort_index()
-            # Контекст/окно детектора берём из cfg.det_* (полный режим), сессия для трейда — из cfg.session_minutes (демо/полный)
-            wins = find_spike_windows(
-                df,
-                context_minutes=cfg.det_context,
-                window_minutes=cfg.det_window,
-                abs_change_threshold_pct=cfg.det_abs_change_pct,
-                contrast_min=cfg.det_contrast_min,
-                cooldown_minutes=cfg.det_cooldown,
-                use_lookahead=cfg.det_use_lookahead,
-            )
+            df = _ensure_utc_index(feed[sym]).sort_index()
+            # --- Построение окон: режим "sliding" (каждый минутный бар) или "spike" (детектор всплесков) ---
+            if getattr(cfg, "index_mode", "spike") == "sliding":
+                wins = _build_sliding_windows(
+                    df=df,
+                    ctx_minutes=cfg.ctx_minutes,
+                    session_minutes=cfg.session_minutes,
+                    stride_minutes=max(1, int(getattr(cfg, "sliding_stride_minutes", 1))),
+                )
+            else:
+                wins = find_spike_windows(
+                    df,
+                    context_minutes=cfg.det_context,
+                    window_minutes=cfg.det_window,
+                    abs_change_threshold_pct=cfg.det_abs_change_pct,
+                    contrast_min=cfg.det_contrast_min,
+                    cooldown_minutes=cfg.det_cooldown,
+                    use_lookahead=cfg.det_use_lookahead,
+                )
@@
         print(f"Index saved: {cfg.index_csv}  | total windows: {total_wins}")
     else:
         idx = pd.read_csv(cfg.index_csv, parse_dates=["ctx_start","ctx_end","session_start","session_end"])
+
+def _build_sliding_windows(
+    df: pd.DataFrame,
+    ctx_minutes: int,
+    session_minutes: int,
+    stride_minutes: int = 1,
+) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
+    """
+    Строит эпизоды по КАЖДОЙ минуте с шагом stride_minutes.
+    Возвращает список (ctx_start, ctx_end, ses_start, ses_end, abs_change_pct=nan).
+    """
+    if df.empty:
+        return []
+    df = _ensure_utc_index(df).sort_index()
+    ts = df.index.unique().sort_values()
+    wins: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, float]] = []
+    # требуем полное покрытие минут в [ctx_start, ses_end)
+    for i in range(ctx_minutes, len(ts) - session_minutes, stride_minutes):
+        ses_start = ts[i]
+        ctx_start = ses_start - pd.Timedelta(minutes=ctx_minutes)
+        ctx_end   = ses_start
+        ses_end   = ses_start + pd.Timedelta(minutes=session_minutes)
+        full_range = pd.date_range(ctx_start, ses_end - pd.Timedelta(minutes=1), freq="T", tz="UTC")
+        slice_df = df.loc[(df.index >= full_range[0]) & (df.index <= full_range[-1])]
+        if len(slice_df) == len(full_range):
+            wins.append((ctx_start, ctx_end, ses_start, ses_end, float("nan")))
+    return wins
*** End Patch
```

```diff
*** Begin Patch
*** Update File: third_party/rl-trading-binance/configs/alpha.py
@@
 data = {
     "source": "stream_sim_db",
     "time_range": {"start_utc": "2025-03-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"},
     # Базовый (демо) режим: 30-10 — полная совместимость с README (Demo) :contentReference[oaicite:10]{index=10}
     "ctx_minutes": 30,
     "session_minutes": 10,
+    # Новый режим индекса: скользящее окно на КАЖДОМ минутном баре
+    "index_mode": "sliding",
+    "sliding_stride_minutes": 1,
@@
     "build_index_from_db": True,
@@
     "symbols": ["OMUSDT"],
@@
         "abs_change_pct": 1.0,       # |ΔP| over window, %
         "contrast_min": 9.0,         # (|ΔP| / avg_abs_ret_pre) ≥ contrast_min
         "cooldown_minutes": 30
     },
*** End Patch
```

---

## Что изменилось и где

* Добавлен переключатель режима построения индекса **ровно в том месте**, где раньше всегда вызывался детектор всплесков `find_spike_windows(...)`. Теперь при `index_mode="sliding"` берётся **каждый минутный бар** как старт сессии, при полном покрытии минут контекста и сессии. 
* Парсинг новых полей (`index_mode`, `sliding_stride_minutes`) и их хранение в `Cfg` — в `_load_cfg(...)` и `dataclass Cfg`. 
* Конфиг `alpha.py` получил строки `"index_mode": "sliding"` и `"sliding_stride_minutes": 1` (остальные параметры оставлены без изменений). 

---

## Чек-лист запуска

| Шаг | Действие                                                                                                           | KPI/риск                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| 1   | Применить патч (2 файла)                                                                                           | Активируется режим скользящих окон                                                                      |
| 2   | Убедиться, что `build_index_from_db=True` и `symbols` заданы                                                       | Индекс сформируется заново                                                                              |
| 3   | Запустить: `python third_party/rl-trading-binance/paper_trader.py third_party/rl-trading-binance/configs/alpha.py` | В логе увидите `Starting index generation...` и намного больше окон (каждая минута при полном покрытии) |
| 4   | Если минутки в БД имеют дыры — окна будут пропускаться (проверка покрытия сохраняется)                             | При необходимости заполнить пропуски данных                                                             |

---

Если нужно, добавлю опцию логирования пропусков покрытия (`coverage_gap`) отдельным PR — без изменения логики.
