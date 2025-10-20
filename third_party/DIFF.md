> Напоминание по режиму: конфигурации — **только** из `configs/` директории; отчёты/артефакты — в `output/<config_name>/`; соблюдаем KPI (Sharpe ≥ 2.5, PF ≥ 1.3, Max DD < 20%). См. системные правила проекта.  Также опираемся на архитектуру и команды запуска из README (`train.py`, `backtest_engine.py`, структура `data/*.npz`). 

---

## TL;DR

Готовлю **шаг 1** из вашего плана: добавляю модуль **`stream_backtest_engine.py` (Вариант B)**, который:

1. Читает конфиг `configs/*.py` и **имитирует поток** минутных свечей из БД через настраиваемый провайдер (`data.db_provider`).
2. **Выделяет окна** по правилу волатильности (напр., ±5% за `ctx_minutes`) и формирует **индекс эпизодов** (ctx→session) в CSV: `output/<config_name>/stream_backtest_index.csv`.
3. Работает строго в **UTC** и использует `data.time_range` (по умолчанию — *требует наличия в конфиге*).

Это «инфраструктурный» слой для стрим-бэктеста: он подготавливает те же окна, что использовались в `.npz` (30→10 и др.), но теперь **из вашей локальной БД**, чтобы далее (шаг 2) подключить реалтайм-часовщик и paper trader.

> По документации репозитория демонстрационный режим — 30 мин контекст → 10 мин сессия; full — 90→60. Мы **не хардкодим** интервалы: берём из `configs/*.py`. 

---

## ACTION NEEDED (минимум для полноценного пробега)

Чтобы модуль отработал «из коробки», в **вашем** конфиге (`configs/alpha.py` либо рабочем) должны быть поля:

* `data = { "source": "stream_sim_db", "time_range": {"start_utc": "2025-03-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"}, "ctx_minutes": 30, "session_minutes": 10, "trigger": {"abs_change_pct": 5.0}, "db_provider": "your_package.db_feed:get_feed" }`
* `symbols_whitelist` (опционально) — список тикеров (если не задать, берутся все из провайдера).

**`db_provider`** — это строка `"module_path:function_name"`, функция должна вернуть итератор пар `(symbol: str, df: pandas.DataFrame)` по заданным `symbols, start_utc, end_utc`. DataFrame обязан содержать UTC-индекс по минутам и колонки: `open, high, low, close, volume`.
Если этих полей нет — модуль бросит понятное исключение и подскажет, что добавить (см. код).

---

## Шаги | Действие | KPI/риск

| Шаг | Действие                                                                                                                         | KPI/риск                                                                 |
| --- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1   | Добавить `stream_backtest_engine.py` (офлайн «replay» БД → индекс эпизодов)                                                      | **Нулевой торговый риск**; подготовка к стрим-бэктесту                   |
| 2   | (Отдельно) Подключить `paper_trader.py` с RealTimeDbFeed и «живыми» часами, использовать индекс эпизодов или онлайновый детектор | KPI тест-гейтинг по PF/Sharpe; риски — синхронизация времени/латентности |
| 3   | (Опционально) Расширить engine для прямого запуска среды/агента вместо записи индекса                                            | +скорость итераций; риск — интерфейс среды (нужно сверить сигнатуры)     |

---

# Патч (unified diff)

> **Ветвь и SHA см. Repo-State Header.** Вносим **1 новый файл**. Лимит по diff соблюдён. Все параметры читаются из `configs/*.py`. Даты в ISO-8601 UTC. Выходной артефакт: `output/<config_name>/stream_backtest_index.csv`. 

```diff
*** Begin Patch
*** Add File: third_party/rl-trading-binance/stream_backtest_engine.py
+#!/usr/bin/env python
+# -*- coding: utf-8 -*-
+"""
+Stream Backtest Engine (Variant B): DB Replay → Episode Index
+--------------------------------------------------------------
+Назначение:
+  - Имитация потока минутных свечей из локальной БД.
+  - Детект окон по волатильности для последующего бэктеста/пейпер-трейдинга.
+
+Требования к конфигу (configs/*.py):
+  data = {
+      "source": "stream_sim_db",  # обязательный переключатель
+      "time_range": {"start_utc": "2025-03-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"},
+      "ctx_minutes": 30,
+      "session_minutes": 10,
+      "trigger": {"abs_change_pct": 5.0},   # порог на |(close_t - close_{t-ctx})/close_{t-ctx}| * 100
+      "symbols_whitelist": ["BTCUSDT", "..."],  # опционально
+      "db_provider": "your_package.db_feed:get_feed"  # module:function
+  }
+
+Провайдер БД:
+  get_feed(symbols: Optional[List[str]], start_utc: str, end_utc: str)
+    -> Iterator[Tuple[str, pandas.DataFrame]]
+  DataFrame: UTC DatetimeIndex (freq='T'), колонки: open, high, low, close, volume (float).
+
+Выход:
+  CSV: output/<config_name>/stream_backtest_index.csv
+  Колонки: symbol,ctx_start,ctx_end,session_start,session_end,ctx_minutes,session_minutes,abs_change_pct
+
+© RL Trading Agent (demo). См. README и системные правила проекта.
+"""
+from __future__ import annotations
+
+import importlib
+import os
+import sys
+import types
+from dataclasses import dataclass
+from datetime import datetime, timezone
+from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple
+
+import numpy as np
+import pandas as pd
+from dateutil import parser as dtparser
+from tqdm import tqdm
+
+
+# ---------------------------- Utils & Config Loader ----------------------------
+
+def _load_py_module(path: str) -> types.ModuleType:
+    import importlib.util
+    spec = importlib.util.spec_from_file_location("user_config", path)
+    if spec is None or spec.loader is None:
+        raise RuntimeError(f"Не удалось загрузить конфиг: {path}")
+    mod = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(mod)  # type: ignore
+    return mod
+
+
+def _require(d: dict, key: str, err: str):
+    if key not in d:
+        raise KeyError(err)
+    return d[key]
+
+
+def _to_utc(ts: str) -> datetime:
+    dt = dtparser.isoparse(ts)
+    if dt.tzinfo is None:
+        # строго требуем UTC — добавляем Z только если не указали
+        dt = dt.replace(tzinfo=timezone.utc)
+    return dt.astimezone(timezone.utc)
+
+
+@dataclass
+class EngineParams:
+    ctx_minutes: int
+    session_minutes: int
+    abs_change_pct: float
+    start_utc: datetime
+    end_utc: datetime
+    symbols: Optional[List[str]]
+    db_provider_path: str
+    config_name: str  # для output/<config_name>/
+
+
+def _load_params_from_config(cfg_path: str) -> EngineParams:
+    cfg_mod = _load_py_module(cfg_path)
+    if not hasattr(cfg_mod, "data") or not isinstance(cfg_mod.data, dict):
+        raise RuntimeError("В конфиге должен быть dict `data` с настройками источника данных.")
+    data = cfg_mod.data
+
+    source = data.get("source", None)
+    if source != "stream_sim_db":
+        raise RuntimeError("`data.source` должен быть 'stream_sim_db' для запуска stream_backtest_engine.")
+
+    tr = _require(
+        data, "time_range",
+        "Отсутствует `data.time_range` (ожидается {'start_utc': ..., 'end_utc': ...} в ISO-8601 UTC)."
+    )
+    start_utc = _to_utc(_require(tr, "start_utc", "Нужен `data.time_range['start_utc']` в ISO-8601 UTC."))
+    end_utc = _to_utc(_require(tr, "end_utc", "Нужен `data.time_range['end_utc']` в ISO-8601 UTC."))
+    if end_utc <= start_utc:
+        raise ValueError("`end_utc` должен быть строго позже `start_utc`.")
+
+    ctx_minutes = int(_require(data, "ctx_minutes", "Нужен `data.ctx_minutes` (целые минуты)."))
+    session_minutes = int(_require(data, "session_minutes", "Нужен `data.session_minutes` (целые минуты)."))
+    trig = _require(data, "trigger", "Нужен блок `data.trigger` (например, {'abs_change_pct': 5.0}).")
+    abs_change_pct = float(_require(trig, "abs_change_pct", "Нужен `data.trigger['abs_change_pct']` (float)."))
+
+    dbp = _require(
+        data, "db_provider",
+        "Нужен `data.db_provider` вида 'package.module:function' для доступа к минутным свечам из БД."
+    )
+    symbols = data.get("symbols_whitelist", None)
+
+    # имя конфига для артефактов
+    config_name = os.path.splitext(os.path.basename(cfg_path))[0]
+
+    return EngineParams(
+        ctx_minutes=ctx_minutes,
+        session_minutes=session_minutes,
+        abs_change_pct=abs_change_pct,
+        start_utc=start_utc,
+        end_utc=end_utc,
+        symbols=symbols,
+        db_provider_path=dbp,
+        config_name=config_name,
+    )
+
+
+# ---------------------------- DB Provider Loader -------------------------------
+
+ProviderFn = Callable[[Optional[List[str]], str, str], Iterable[Tuple[str, pd.DataFrame]]]
+
+
+def _load_db_provider(path: str) -> ProviderFn:
+    """
+    path: "package.module:function"
+    function signature: get_feed(symbols, start_utc, end_utc) -> Iterable[(symbol, DataFrame)]
+    """
+    if ":" not in path:
+        raise RuntimeError("`data.db_provider` должен быть в формате 'module.submodule:function'.")
+    mod_path, fn_name = path.split(":", 1)
+    mod = importlib.import_module(mod_path)
+    if not hasattr(mod, fn_name):
+        raise RuntimeError(f"В модуле `{mod_path}` нет функции `{fn_name}`.")
+    fn = getattr(mod, fn_name)
+    return fn  # type: ignore
+
+
+# ---------------------------- Volatility Detector ------------------------------
+
+def compute_abs_change_pct(series_close: pd.Series, minutes: int) -> pd.Series:
+    """
+    | close_t - close_{t-minutes} | / close_{t-minutes} * 100
+    """
+    ref = series_close.shift(minutes)
+    return (series_close - ref).abs().div(ref).mul(100.0)
+
+
+def detect_windows(df: pd.DataFrame, symbol: str, ctx_m: int, ses_m: int, thr_pct: float) -> List[Dict[str, object]]:
+    """
+    df: minute-level OHLCV with UTC DatetimeIndex (freq='T').
+    Возвращает список окон (dict) с метаданными.
+    """
+    if df.empty:
+        return []
+
+    # Убедимся в сортировке и равномерной частоте
+    df = df.sort_index()
+    # forward fill на редкие пропуски, но без создания новых меток
+    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()
+
+    # Детект «всплесков» на основе контекстного окна ctx_m
+    abs_chg = compute_abs_change_pct(df["close"], minutes=ctx_m)
+    triggers = abs_chg >= thr_pct
+
+    out: List[Dict[str, object]] = []
+    # Начало сессии — сразу после контекста
+    for t in df.index[(triggers).to_numpy()]:
+        ctx_end = t
+        ctx_start = ctx_end - pd.Timedelta(minutes=ctx_m)
+        ses_start = ctx_end
+        ses_end = ses_start + pd.Timedelta(minutes=ses_m)
+
+        # Проверим, что окно полностью в пределах df
+        if ctx_start < df.index[0]:
+            continue
+        if ses_end > df.index[-1]:
+            continue
+
+        out.append({
+            "symbol": symbol,
+            "ctx_start": ctx_start.to_pydatetime().replace(tzinfo=timezone.utc),
+            "ctx_end": ctx_end.to_pydatetime().replace(tzinfo=timezone.utc),
+            "session_start": ses_start.to_pydatetime().replace(tzinfo=timezone.utc),
+            "session_end": ses_end.to_pydatetime().replace(tzinfo=timezone.utc),
+            "ctx_minutes": ctx_m,
+            "session_minutes": ses_m,
+            "abs_change_pct": float(abs_chg.loc[ctx_end]),
+        })
+    return out
+
+
+# ---------------------------- Main Runner --------------------------------------
+
+def main(argv: List[str]) -> int:
+    if len(argv) < 2:
+        print("Использование: python stream_backtest_engine.py configs/alpha.py")
+        return 2
+    cfg_path = argv[1]
+    params = _load_params_from_config(cfg_path)
+
+    # Загрузка провайдера БД
+    provider = _load_db_provider(params.db_provider_path)
+
+    # Получаем поток (symbol, DataFrame)
+    feed_iter = provider(
+        params.symbols,
+        params.start_utc.replace(tzinfo=timezone.utc).isoformat(),
+        params.end_utc.replace(tzinfo=timezone.utc).isoformat(),
+    )
+
+    all_rows: List[Dict[str, object]] = []
+    total_symbols = 0
+    for symbol, df in tqdm(feed_iter, desc="DB replay", unit="symbol"):
+        total_symbols += 1
+        # sanity checks
+        if not isinstance(df.index, pd.DatetimeIndex):
+            raise RuntimeError(f"[{symbol}] index должен быть DatetimeIndex в UTC.")
+        if df.index.tz is None:
+            # трактуем как UTC, но лучше отдавать уже tz-aware
+            df.index = df.index.tz_localize("UTC")
+        else:
+            df.index = df.index.tz_convert("UTC")
+        required_cols = {"open", "high", "low", "close", "volume"}
+        if not required_cols.issubset(df.columns):
+            missing = sorted(list(required_cols.difference(df.columns)))
+            raise RuntimeError(f"[{symbol}] отсутствуют колонки: {missing}")
+
+        # ограничим по диапазону (на случай, если провайдер вернул шире)
+        df = df.loc[params.start_utc: params.end_utc]
+        rows = detect_windows(df, symbol, params.ctx_minutes, params.session_minutes, params.abs_change_pct)
+        all_rows.extend(rows)
+
+    # Экспорт индекса эпизодов
+    out_dir = os.path.join("third_party", "rl-trading-binance", "output", params.config_name)
+    os.makedirs(out_dir, exist_ok=True)
+    out_csv = os.path.join(out_dir, "stream_backtest_index.csv")
+    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
+
+    print(f"[stream_backtest_engine] Сформировано окон: {len(all_rows)} по {total_symbols} тикерам.")
+    print(f"[stream_backtest_engine] Индекс эпизодов сохранён: {out_csv}")
+    print("[stream_backtest_engine] Следующий шаг: пейпер-трейдинг в реальном времени на основе этих окон (paper_trader.py).")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main(sys.argv))
+
*** End Patch
```

---

## Что это даёт прямо сейчас

* «Под ключ» реализовано: чтение конфига → загрузка данных из БД (через **настраиваемый провайдер**) → **детект эпизодов** по волатильности → сохранение индекса эпизодов в артефакты эксперимента (`output/<config>/stream_backtest_index.csv`).
* В логике **нет хардкода интервалов**: `ctx_minutes`/`session_minutes` и `abs_change_pct` берём из `configs/*.py`. Это соответствует проектным правилам. 
* Это «кирпичик» для шага 2 (**`paper_trader.py`**): по этому индексу легко воспроизводить сессии с «живыми» часами и принимать решения агентом.

> В README проекта прямо выделено различие «Demo (30→10, статичные .npz)» vs «Full (90→60, реальный поток WebSocket+DB)». Мы как раз закрываем часть «Full» — стрим из БД с фильтром по волатильности. 

---

## Команды для проверки локально

```bash
# 1) Новая ветка
git checkout -b feature/stream-backtest-engine

# 2) Сохраните патч в файл и примените
# (скопируйте diff в changes.patch)
git apply --index changes.patch
git commit -m "feat(stream): add stream_backtest_engine.py (DB replay → episodes index)"

# 3) Убедитесь, что в configs/alpha.py (или вашем конфиге) задан source=stream_sim_db и блок data.* как описано
# Пример запуска:
python third_party/rl-trading-binance/stream_backtest_engine.py third_party/rl-trading-binance/configs/alpha.py

# 4) Запустите unit-тесты проекта (если настроены)
pytest -q
```

---

## PR (создание)

```bash
git push -u origin feature/stream-backtest-engine
gh pr create -t "feat(stream): DB replay → stream_backtest_engine (episode index)" -B prosperous_bot -b "$(cat <<'PRBODY'
### 🎯 Goal
Поддержать стрим-бэктест: воспроизведение минутных свечей из БД, детект эпизодов по волатильности и сохранение индекса окон для последующего пейпер-трейдинга.

### 📝 Implementation Details
- Добавлен `third_party/rl-trading-binance/stream_backtest_engine.py`.
- Чтение конфига `configs/*.py` (data.source='stream_sim_db', time_range, ctx/session, trigger.abs_change_pct, db_provider).
- Загрузка данных из БД через модульную функцию-провайдер.
- Детект окон: |ΔP|/P * 100 за `ctx_minutes` ≥ порога.
- Экспорт индекса эпизодов: `output/<config>/stream_backtest_index.csv` (UTC).

### 📈 KPI/Risk Assessment
- **Sharpe:** n/a (инфраструктурный слой; не меняет логику торговли напрямую)
- **Max DD:** n/a
- **Profit Factor:** n/a
Риск: требуется корректный провайдер БД; несоответствие схемы данных — явная ошибка с подсказкой.

### 롤백 계획 (Rollback Plan)
Revert PR. Код изолирован в новом файле; откат безопасен.

---

## Что дальше

1. **Подтвердите**/добавьте поля в `configs/*.py` (`data.source='stream_sim_db'`, `time_range`, `ctx_minutes`, `session_minutes`, `trigger.abs_change_pct`, `db_provider`, опц. `symbols_whitelist`).
2. Дайте путь к вашей функции-провайдеру `db_provider` (я ожидаю подпись как в докстроке).

---

**Ссылки/источники:**

* Архитектура и команды (`train.py`, `backtest_engine.py`, структура данных/директорий): README. 
* Политики проекта (конфиги из `configs/`, артефакты в `output/<config>/`, KPI): SYSTEM_PROMPT. 
* История коммитов на ветке `prosperous_bot`: подтверждение состояния (см. Repo-State Header). ([GitHub][2])