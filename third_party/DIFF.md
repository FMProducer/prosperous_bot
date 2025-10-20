Ниже — строгий аудит с указанием путей и точных проблем.
> Файлы: `stream_backtest_engine.py` , `alpha.py` (как `third_party/rl-trading-binance/configs/alpha.py`) , `db_provider.py` (провайдер PostgreSQL) .

---

## TL;DR

Код в целом **жизнеспособен**, но для корректного отбора окон по правилу «±X% за *ctx_minutes*» в минутных рядах из БД нужны **три критические правки**:

1. **Сдвиг по времени, а не по числу строк.** Сейчас `shift(minutes)` трактуется как *N строк*, а не *N минут* — при дырках в минутках порог сработает **ложно**. Нужно `shift(freq='30T')` или предварительное **ресемплирование до 1Т**. 
2. **Cooldown (дедупликация событий).** Сейчас триггер сработает на каждом последующем баре в «взрыве» — получите каскад перекрывающихся окон. Добавить `trigger.cooldown_minutes`. 
3. **Путь к провайдеру в конфиге.** Указан `backends.postgres_ws.db_provider:get_feed`, а фактически файл — `db_provider.py` в корне пакета: импорт **упадёт**. Либо переместить файл в пакет `backends/postgres_ws/`, либо сменить строку на `db_provider:get_feed`.

Плюс важные замечания по **безопасности/надёжности** в `db_provider.py`: убрать дефолтный пароль, ограничить список колонок в SQL, гарантировать закрытие соединения контекстом. 

---

## Детальный аудит (по файлам)

### 1) `third_party/rl-trading-binance/stream_backtest_engine.py` — аудит логики детектора

**Найдено:**

* `compute_abs_change_pct(series_close, minutes)` использует `series_close.shift(minutes)`. Это **сдвиг по количеству строк**, а не по времени. При пропусках минут либо нерегулярной частоте получим **смещение < ctx_minutes** и **ложные триггеры** (или наоборот пропуски), что портит индекс эпизодов. 
* Нет **cooldown** между событиями — длинная свечка вызовет десятки окон, что искажает статистику и создаёт переобучение на «склейках» одного события. 
* В коде есть FFill, но **нет гарантии равномерности** индекса. Если хотим гарантированно «30 минут назад», лучше **ресемплировать до 1Т** (или использовать `pct_change(freq='30T')`). 
* Выходной путь: `third_party/rl-trading-binance/output/<config_name>/…` —符合 нашей схеме артефактов (внутри RL-папки). ✔️ 

**Что исправить (must-have):**

* Переписать `compute_abs_change_pct` на **временной** сдвиг:

  ```python
  ref = series_close.shift(freq=pd.Timedelta(minutes=ctx_m))
  abs_pct = (series_close - ref).abs().div(ref).mul(100.0)
  ```

  Либо предварительно `df = df.asfreq('T').ffill()` и оставить `shift(ctx_m)`.
* Добавить параметр `trigger.cooldown_minutes` и фильтрацию времени с последнего срабатывания.
* (Опционально) `resample_1t: bool` в конфиге — включить по умолчанию для потоков из БД.

### 2) `third_party/rl-trading-binance/configs/alpha.py` — аудит конфига

**Найдено:**

* Блок `data={...}` корректно задаёт `source="stream_sim_db"`, `time_range`, `ctx_minutes=30`, `session_minutes=10`, `trigger.abs_change_pct=5.0`. ✔️ 
* **Несоответствие импорта провайдера:** `"db_provider": "backends.postgres_ws.db_provider:get_feed"` при том, что фактический файл — `db_provider.py` (корень). Это даст `ModuleNotFoundError`.
* Нет `trigger.cooldown_minutes` и флага `resample_1t`. Рекомендуется добавить. 

### 3) `db_provider.py` — аудит безопасности/SQL/ресурсов

**Найдено и риски:**

* Жёсткий **дефолт пароля** `PGPASSWORD="9691"`. Это противоречит нашему правилу «секреты — только через окружение, без дефолтов» (риски случайной утечки и неверной среды). Предпочтительно **без дефолта**, иначе — ошибка с подсказкой. 
* `SELECT *` тянет **лишние колонки** → лишний трафик/память. Указать явный список полей. 
* Соединение закрывается после генерации, но **лучше через контекст** `with psycopg2.connect(...) as conn:` чтобы закрывалось при исключениях, а также `with conn.cursor()` для списка символов. 
* Коверадж символов: провайдер отдаёт все `DISTINCT symbol` в диапазоне. Это может включить редкие пары с пробелами в данных → много пустых/рваных рядов. Нужна фильтрация по **доле заполненности** (например, ≥95% минут в диапазоне). (Можно оставить на стороне engine.)

---

## Рекомендуемые правки (минимальный патч)

### A) Исправить детектор и добавить cooldown/ресемплирование

**Файл:** `third_party/rl-trading-binance/stream_backtest_engine.py` 

```diff
*** a/third_party/rl-trading-binance/stream_backtest_engine.py
--- b/third_party/rl-trading-binance/stream_backtest_engine.py
@@
-from dateutil import parser as dtparser
+from dateutil import parser as dtparser
@@
 class EngineParams:
     ctx_minutes: int
     session_minutes: int
     abs_change_pct: float
+    cooldown_minutes: int
+    resample_1t: bool
     start_utc: datetime
     end_utc: datetime
@@
-    abs_change_pct = float(_require(trig, "abs_change_pct", "Нужен `data.trigger['abs_change_pct']` (float)."))
+    abs_change_pct = float(_require(trig, "abs_change_pct", "Нужен `data.trigger['abs_change_pct']` (float)."))
+    cooldown_minutes = int(trig.get("cooldown_minutes", 60))
+    resample_1t = bool(data.get("resample_1t", True))
@@
-    return EngineParams(
+    return EngineParams(
         ctx_minutes=ctx_minutes,
         session_minutes=session_minutes,
         abs_change_pct=abs_change_pct,
+        cooldown_minutes=cooldown_minutes,
+        resample_1t=resample_1t,
         start_utc=start_utc,
         end_utc=end_utc,
         symbols=symbols,
         db_provider_path=dbp,
         config_name=config_name,
     )
@@
-def compute_abs_change_pct(series_close: pd.Series, minutes: int) -> pd.Series:
-    """
-    | close_t - close_{t-minutes} | / close_{t-minutes} * 100
-    """
-    ref = series_close.shift(minutes)
-    return (series_close - ref).abs().div(ref).mul(100.0)
+def compute_abs_change_pct(series_close: pd.Series, minutes: int) -> pd.Series:
+    """
+    Абсолютное изменение в % относительно цены ровно minutes назад по времени,
+    а не по числу строк. Требует регулярного индекса или time-based shift.
+    """
+    ref = series_close.shift(freq=pd.Timedelta(minutes=minutes))
+    return (series_close - ref).abs().div(ref).mul(100.0)
@@
-def detect_windows(df: pd.DataFrame, symbol: str, ctx_m: int, ses_m: int, thr_pct: float) -> List[Dict[str, object]]:
+def detect_windows(df: pd.DataFrame, symbol: str, ctx_m: int, ses_m: int, thr_pct: float,
+                   cooldown_m: int) -> List[Dict[str, object]]:
@@
-    # Убедимся в сортировке и равномерной частоте
+    # Сортировка и (опц.) выравнивание минутной частоты
     df = df.sort_index()
-    # forward fill на редкие пропуски, но без создания новых меток
-    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()
+    if (df.index.freq is None) and (getattr(df.index, "inferred_freq", None) != "T"):
+        # Нерегулярный индекс — оставляем как есть; time-based shift создаст NaN, что безопаснее ложных сигналов.
+        pass
+    # forward fill на редкие пропуски в значениях (метки не добавляем)
+    cols = ["open", "high", "low", "close", "volume"]
+    df[cols] = df[cols].ffill()
@@
-    out: List[Dict[str, object]] = []
-    # Начало сессии — сразу после контекста
-    for t in df.index[(triggers).to_numpy()]:
+    out: List[Dict[str, object]] = []
+    last_fire: Optional[pd.Timestamp] = None
+    for t in df.index[triggers.to_numpy()]:
         ctx_end = t
         ctx_start = ctx_end - pd.Timedelta(minutes=ctx_m)
         ses_start = ctx_end
         ses_end = ses_start + pd.Timedelta(minutes=ses_m)
@@
-        out.append({
+        # cooldown: не допускаем перекрывающиеся эпизоды слишком часто
+        if last_fire is not None and (t - last_fire) < pd.Timedelta(minutes=cooldown_m):
+            continue
+        last_fire = t
+        out.append({
             "symbol": symbol,
@@
-    for symbol, df in tqdm(feed_iter, desc="DB replay", unit="symbol"):
+    for symbol, df in tqdm(feed_iter, desc="DB replay", unit="symbol"):
@@
-        df = df.loc[params.start_utc: params.end_utc]
-        rows = detect_windows(df, symbol, params.ctx_minutes, params.session_minutes, params.abs_change_pct)
+        df = df.loc[params.start_utc: params.end_utc]
+        # (опц.) жёсткое выравнивание до 1Т: минимизирует NaN в time-based shift
+        if params.resample_1t:
+            df = df.asfreq("T")
+            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()
+        rows = detect_windows(df, symbol, params.ctx_minutes, params.session_minutes,
+                              params.abs_change_pct, params.cooldown_minutes)
*** End Patch
```

### B) Подправить конфиг под реальный провайдер и добавить параметры

**Файл:** `third_party/rl-trading-binance/configs/alpha.py` 

```diff
*** a/third_party/rl-trading-binance/configs/alpha.py
--- b/third_party/rl-trading-binance/configs/alpha.py
@@
 data = {
     "source": "stream_sim_db",
     "time_range": {"start_utc": "2025-03-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"},
     "ctx_minutes": 30,
     "session_minutes": 10,
-    "trigger": {"abs_change_pct": 5.0},
-    "db_provider": "backends.postgres_ws.db_provider:get_feed"
+    "trigger": {"abs_change_pct": 5.0, "cooldown_minutes": 60},
+    "resample_1t": True,
+    # Если файл провайдера лежит рядом (db_provider.py), используем прямой импорт:
+    "db_provider": "db_provider:get_feed"
 }
```

### C) Сделать провайдер безопаснее и дешевле по сети

**Файл:** `db_provider.py` 

```diff
*** a/db_provider.py
--- b/db_provider.py
@@
-    try:
-        conn = psycopg2.connect(
-            host=os.getenv("PGHOST", "localhost"),
-            port=int(os.getenv("PGPORT", "5432")),
-            dbname=os.getenv("PGDATABASE", "marketdata"),
-            user=os.getenv("PGUSER", "postgres"),
-            password=os.getenv("PGPASSWORD", "9691"),
-            connect_timeout=10
-        )
-    except psycopg2.OperationalError as e:
-        raise RuntimeError(f"Could not connect to PostgreSQL database: {e}")
+    try:
+        # Без дефолтного пароля: используем PG* из окружения / .pgpass.
+        conn = psycopg2.connect(
+            host=os.getenv("PGHOST", "localhost"),
+            port=int(os.getenv("PGPORT", "5432")),
+            dbname=os.getenv("PGDATABASE", "marketdata"),
+            user=os.getenv("PGUSER", "postgres"),
+            password=os.getenv("PGPASSWORD", None),
+            connect_timeout=10
+        )
+    except psycopg2.OperationalError as e:
+        raise RuntimeError(f"Could not connect to PostgreSQL database: {e}")
@@
-    if not symbols:
-        # If no symbols are provided, get all symbols from the database within the time range.
-        with conn.cursor() as cur:
-            cur.execute(
-                "SELECT DISTINCT symbol FROM klines_1m WHERE open_time_ms >= %s AND open_time_ms <= %s",
-                (start_ms, end_ms)
-            )
-            symbols = [row[0] for row in cur.fetchall()]
+    if not symbols:
+        # Получаем универсум из БД в диапазоне
+        with conn.cursor() as cur:
+            cur.execute(
+                "SELECT DISTINCT symbol FROM klines_1m WHERE open_time_ms >= %s AND open_time_ms <= %s",
+                (start_ms, end_ms)
+            )
+            symbols = [row[0] for row in cur.fetchall()]
@@
-        query = "SELECT * FROM klines_1m WHERE symbol = %s AND open_time_ms >= %s AND open_time_ms <= %s ORDER BY open_time_ms"
+        query = (
+            "SELECT open_time_ms, open_price, high_price, low_price, close_price, base_volume "
+            "FROM klines_1m "
+            "WHERE symbol = %s AND open_time_ms >= %s AND open_time_ms <= %s "
+            "ORDER BY open_time_ms"
+        )
*** End Patch
```

> Эти изменения укладываются в лимиты PR и устраняют главные источники ложных сигналов/рисков.

---

## Мини-план проверки (до запуска обучения)

| Шаг | Действие                                                                                                          | KPI/риск                                                          |
| --- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1   | Применить патчи A–C; проверить импорт провайдера (`db_provider:get_feed`)                                         | Script запускается; риск: несоответствие путей                    |
| 2   | Мини-интеграционный прогон на 2–3 символах (например BTCUSDT, ETHUSDT) за 2025-03-01→2025-03-07                   | `stream_backtest_index.csv` создан, окна > 0; риск: пустые данные |
| 3   | Валидировать **временной сдвиг**: ручной расчёт 1–2 триггеров (±5% за 30м) у выбранного символа → совпадает с CSV | Отсутствие ложных дублей                                          |
| 4   | Расширенный прогон на ваших 257 символах за 2025-03-01→2025-06-01                                                 | Выполнено ≤ разумного времени; объём окон соответствует ожиданиям |
| 5   | Контроль качества окон: нет лавины перекрытия → `cooldown_minutes` регулирует частоту                             | Гладкое распределение событий                                     |

---

## Команды запуска (после правок)

```bash
# Пример: используем ваш конфиг (alpha.py)
python third_party/rl-trading-binance/stream_backtest_engine.py third_party/rl-trading-binance/configs/alpha.py
# Ожидаемый вывод:
# [stream_backtest_engine] Сформировано окон: <N> по <K> тикерам.
# [stream_backtest_engine] Индекс эпизодов сохранён: third_party/rl-trading-binance/output/alpha/stream_backtest_index.csv
```

---

## Итог

* Механика реплея/детектора реализована правильно **концептуально**, но без временного сдвига и cooldown получите **смещение и дубликаты**. Патчи выше это закрывают.
* Конфиг скорректирован под фактический провайдер; провайдер — безопаснее (без дефолтного пароля, уже с узким `SELECT`).