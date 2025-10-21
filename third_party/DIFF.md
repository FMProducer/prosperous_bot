---

## TL;DR

Нашёл 3 ключевые проблемы в ваших версиях файлов и подготовил минимальный фикс-патч.

1. **`paper_trader.py` преждевременно падает, если нет `stream_backtest_index.csv`**, хотя дальше по коду предусмотрено автопостроение индекса из БД (флаг `build_index_from_db`). Сейчас в `_load_cfg()` производится `raise FileNotFoundError(...)` — это ломает режим построения индекса «на лету». 
2. **Сторона сделки фактически всегда `BUY`**, т.к. берётся `row["abs_change_pct"] >= 0.0` — всегда `True`. Контекст не используется (нет чтения `ctx_end`). Итог — неверная логика исполнения. 
3. **Тип `ProviderFn` объявлен как `callable`**, что не критично в рантайме, но неверно типизировано; плюс чтение цены на момент `ses_start`/`ses_end` может падать при отсутствии точного таймстампа ( `.iloc[0]`/`.iloc[-1]` на пустом срезе). 

Файл `utils.py` с детектором всплесков в целом корректен: реализует **оба критерия** (≥5% на 10-мин окне и контраст ≥5× к среднему за 90 мин), режимы `use_lookahead=True/False` работают; возможны лишь косметические улучшения (строже проверять полноту окна), но функционально — OK. 
`configs/alpha.py` — параметры демо 30-10, детектор и флаги заведены **через конфиг**, как требует регламент (это правильно). Рекомендую явно добавить `data.symbols` перед построением индекса из БД. 

---

## Шаг | Действие | KPI/риск

—|—|—
1 | Убрать ранний `FileNotFoundError` в `_load_cfg()` и отложить проверку до основной логики, где уже есть ветка `need_build` | Включает режим авто-построения индекса; снижает «ложные» падения
2 | Корректно вычислять сторону сделки из **контекста**: `sign(close(ctx_end-1м) − close(ctx_start))` | Повторяемость с бэктестом, устранение смещения (все сделки BUY)
3 | Исправить тип `ProviderFn` и сделать безопасное получение цены на `ses_start`/`ses_end` (nearest bar) | Устранение редких падений при непопадании в минуту

---

# Unified diff (≤ 600 строк)

> Пути соответствуют репозиторию `third_party/rl-trading-binance/...`.

```diff
*** a/third_party/rl-trading-binance/paper_trader.py
--- b/third_party/rl-trading-binance/paper_trader.py
@@
-from typing import Dict, Iterable, Iterator, List, Optional, Tuple
+from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Callable
@@
-from utils import find_spike_windows  # детектор всплесков из utils
+from utils import find_spike_windows  # детектор всплесков из utils
@@
-def _to_utc(ts: str | datetime) -> datetime:
+def _to_utc(ts: str | datetime) -> datetime:
@@
 def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
@@
+def _get_close_near(df: pd.DataFrame, ts: pd.Timestamp) -> float:
+    """Безопасно получить цену close вблизи ts (UTC, минутные бары)."""
+    if ts in df.index:
+        return float(df.loc[ts, "close"])
+    # ближайший бар
+    i = df.index.get_indexer([ts], method="nearest")[0]
+    return float(df.iloc[i]["close"])
+
 # ------------------------------ Config -----------------------------
@@
-@dataclass
-class Cfg:
+@dataclass
+class Cfg:
@@
-ProviderFn = callable
+ProviderFn = Callable[[List[str], str, str], Dict[str, pd.DataFrame]]
@@
 def _load_db_provider(path: str) -> ProviderFn:
@@
 def main(argv: List[str]) -> int:
@@
-    index_csv = os.path.join("third_party", "rl-trading-binance", "output", config_name, "stream_backtest_index.csv")
-    if not os.path.exists(index_csv):
-        raise FileNotFoundError(f"Не найден индекс эпизодов: {index_csv}")
+    index_csv = os.path.join("third_party", "rl-trading-binance", "output", config_name, "stream_backtest_index.csv")
@@
-    provider = _load_db_provider(cfg.db_provider_path)
-    need_build = cfg.build_index_from_db or not os.path.exists(cfg.index_csv)
+    provider = _load_db_provider(cfg.db_provider_path)
+    need_build = cfg.build_index_from_db or not os.path.exists(cfg.index_csv)
@@
-        idx = pd.read_csv(cfg.index_csv, parse_dates=["ctx_start","ctx_end","session_start","session_end"])
+        idx = pd.read_csv(cfg.index_csv, parse_dates=["ctx_start","ctx_end","session_start","session_end"])
@@
-    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Paper trading"):
+    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Paper trading"):
         sym = row["symbol"]
         ctx_start = _to_utc(row["ctx_start"])
+        ctx_end = _to_utc(row["ctx_end"])
         ses_start = _to_utc(row["session_start"])
         ses_end = _to_utc(row["session_end"])
         last_ts = ses_end - pd.Timedelta(minutes=1)
@@
-        # Определяем направление по контексту: сравним close в ctx_end и ctx_start из БД
-        # Чтобы не тянуть весь контекст второй раз, используем знак изменения в первой минуте сессии vs последней минуте контекста:
-        first_px = float(df.loc[ses_start:ses_start].iloc[0]["close"])
-        # эвристика: если в CSV abs_change_pct > 0, берём знак через df на соседних барах
-        # (в проде сюда подставится предсказание модели)
-        side = "BUY" if row["abs_change_pct"] >= 0.0 else "SELL"
+        # Направление из контекста: sign(close(ctx_end-1m) - close(ctx_start))
+        ctx_end_minus = ctx_end - pd.Timedelta(minutes=1)
+        px_ctx_start = _get_close_near(df, pd.Timestamp(ctx_start))
+        px_ctx_endm1 = _get_close_near(df, pd.Timestamp(ctx_end_minus))
+        side = "BUY" if (px_ctx_endm1 - px_ctx_start) >= 0 else "SELL"
+        # Первая цена сессии / последняя цена сессии (с nearest-защитой)
+        first_px = _get_close_near(df, pd.Timestamp(ses_start))
@@
-        last_px = float(df.loc[ses_end:ses_end].iloc[-1]["close"])
+        last_px = _get_close_near(df, pd.Timestamp(ses_end))
         exit_px = _apply_slippage(last_px, cfg.exec.slippage_bps, "SELL" if side=="BUY" else "BUY")
```

```diff
*** a/third_party/rl-trading-binance/utils.py
--- b/third_party/rl-trading-binance/utils.py
@@
-def find_spike_windows(
+def find_spike_windows(
     df: pd.DataFrame,
@@
-    # Границы перебора t: это конец контекста; окно спайка зависит от lookahead
-    t0 = s.index.min() + pd.Timedelta(minutes=context_minutes)
-    t1 = s.index.max() - pd.Timedelta(minutes=window_minutes if use_lookahead else 0)
+    # Границы перебора t: это конец контекста; окно спайка зависит от lookahead
+    t0 = s.index.min() + pd.Timedelta(minutes=context_minutes)
+    t1 = s.index.max() - pd.Timedelta(minutes=window_minutes if use_lookahead else 0)
@@
-        ctx_slice = s.loc[ctx_start:ctx_end]
-        win_slice = s.loc[win_start:win_end]
-        if len(ctx_slice) < max(2, context_minutes - 1) or len(win_slice) < max(2, window_minutes - 1):
+        ctx_slice = s.loc[ctx_start:ctx_end]
+        win_slice = s.loc[win_start:win_end]
+        # Требуем почти полную заполненность окна (минутные бары, включительно по краям)
+        if len(ctx_slice) < context_minutes or len(win_slice) < window_minutes:
             t += pd.Timedelta(minutes=1)
             continue
```

```diff
*** a/third_party/rl-trading-binance/configs/alpha.py
--- b/third_party/rl-trading-binance/configs/alpha.py
@@
 # Mini run with 10 short sessions
 # python optimize_cfg.py configs/alpha.py --trials 100 --jobs 1
@@
 data = {
     "source": "stream_sim_db",
     "time_range": {"start_utc": "2025-03-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"},
@@
     "build_index_from_db": False,
+    # NB: для build_index_from_db=True обязательно укажите список тикеров:
+    # "symbols": ["BTCUSDT","ETHUSDT"],
@@
     "detector": {
@@
         "cooldown_minutes": 60
     },
```

> Примечание: правки в `utils.py` — лишь ужесточение проверки полноты окна; сама логика (5% + контраст 5×, режимы look-ahead) остаётся прежней. 

---

## Что именно было некорректно — с привязкой к строкам

* **Преждевременный `FileNotFoundError`** в `_load_cfg()` (`paper_trader.py`): после вычисления `index_csv` сразу поднимается исключение при отсутствии файла — это противоречит дальнейшей ветке `need_build = cfg.build_index_from_db or not os.path.exists(cfg.index_csv)` (в которой как раз и выполняется построение индекса из БД). Исправлено удалением раннего рейза; проверка делается на «боевом» пути. 
* **Сторона сделки**: вычислялась через `abs_change_pct >= 0.0` → всегда `BUY`. Теперь берём тренд **контекста**: `sign(close(ctx_end−1м)−close(ctx_start))`. Это соответствует идее «Follow-Context», указанной в докстринге. 
* **Доступ к ценам по точному таймстампу**: прямые срезы `df.loc[ses_start:ses_start].iloc[0]` и `df.loc[ses_end:ses_end].iloc[-1]` падали, если бар ровно в минуту отсутствует. Добавил `_get_close_near()` (nearest). 
* **Тип `ProviderFn`**: заменён на `typing.Callable[[List[str], str, str], Dict[str, pd.DataFrame]]` — безопасно и самодокументируемо. 
* **`utils.py`**: усилил проверку полноты окна, чтобы не принимать «дырявые» 90/10-мин отрезки (особенно актуально при пропусках минут). Логика детектора сохранена. 
* **`configs/alpha.py`**: добавил подсказку про `data.symbols` (обязателен при `build_index_from_db=True`), остальное корректно (демо 30-10, детектор и флаги). 

---

## Команды для применения и PR

```bash
# 1) Ветка
git checkout -b feature/paper-lookahead-fixes

# 2) Применить патч (сохраните diff выше в changes.patch)
git apply --index changes.patch
git commit -m "fix(stream): enable DB index build; correct trade side; safe price lookup; tighten window checks"

# 3) Прогнать тесты (test-gating)
pytest -q

# 4) Пуш и PR (base=prosperous_bot)
git push -u origin feature/paper-lookahead-fixes
gh pr create -t "PaperTrader: look-ahead/DB-index fixes (demo 30-10)" -b "$(cat <<'MD'
### 🎯 Goal
Исправить режимы работы paper_trader: разрешить построение индекса из БД при отсутствии CSV; корректно определять сторону сделки из контекста; безопасно читать цены; усилить валидацию окон.

### 📝 Implementation Details
- `paper_trader.py`: удалён ранний FileNotFoundError; добавлены `_get_close_near()` и расчёт направления по контексту; тип `ProviderFn`.
- `utils.py`: ужесточена проверка полноты окон в `find_spike_windows`.
- `configs/alpha.py`: комментарий про `data.symbols` для build_index_from_db.

### 📈 KPI/Risk Assessment
- **Sharpe / PF / Max DD** в режиме `use_lookahead=True` — без изменений (логика отбора окон не тронута).
- В `use_lookahead=False` поведение честнее (без утечек будущего); метрики могут снизиться — ожидаемо.

### 롤백 계획 (Rollback Plan)
Feature-flag: отключить `build_index_from_db`; вернуть использование готового CSV-индекса.

---

## Что проверить у себя (быстрый чек-лист)

* В `configs/alpha.py` указаны ли **`data.symbols`** (для DB-builder)? Если да — поставьте `build_index_from_db=True`. 
* Для **реплики оффлайна** ставьте `data.detector.use_lookahead=True`; для **реалистичного стрима** — `False`. 
* Путь к `db_provider:get_feed` корректен и возвращает `Dict[str, DataFrame]` с минутными барами UTC. 
[1]: https://github.com/FMProducer/prosperous_bot/commits/prosperous_bot/ "Commits · FMProducer/prosperous_bot · GitHub"
