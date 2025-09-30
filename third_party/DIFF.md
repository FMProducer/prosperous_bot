## TL;DR

Я прочитал ваш архив `db_schema_snapshot.zip` и зафиксировал фактическую схему PostgreSQL: две таблицы `public.klines_1m` и `public.agg_trades`, без PK/констрейнтов, но с уникальными индексами на ключах идемпотентности. Для корректировки SQL/скрипта выгрузки в NPZ предлагаю ввести материализованное представление `public.mv_candles_prepared` (минутные свечи в нормализованном виде) + индекс и `REFRESH ... CONCURRENTLY`. Ниже — точные DDL/DML, PowerShell-команды для съёма схемы и минимальный Python-экспортёр (без хардкода, всё через конфиг). Используем `to_timestamp(open_time_ms/1000.0)` и `AT TIME ZONE 'UTC'` корректно согласно документации PostgreSQL ([PostgreSQL][3]). Для `REFRESH CONCURRENTLY` на MV требуется уникальный индекс, что также отражено в оф. доках/разборах ([PostgreSQL][4]).

---

## Что именно у вас в БД (по снапшоту)

* Схемы: `public`
* Таблицы:

  * `public.klines_1m` (свечи 1m): `symbol text`, `open_time_ms bigint`, `open_price numeric`, `high_price numeric`, `low_price numeric`, `close_price numeric`, `base_volume numeric`, `quote_volume numeric`, `trade_count int`, `taker_base numeric`, `taker_quote numeric`, `is_closed boolean`, `ingest_ts timestamptz`.
    Индексы: уникальный на `(symbol, open_time_ms)` и btree на `(open_time_ms)`.
  * `public.agg_trades`: агрегированные трейды; уникальный индекс на `(symbol, agg_id)`.
* Констрейнтов (PK/UK) формально нет — только индексы (это ок для производительности, но с точки зрения семантики лучше добавить `UNIQUE`/`PRIMARY KEY` — см. отличия PK/UNIQUE в доках) ([PostgreSQL][5]).

---

## Рекомендуемая нормализация под выгрузку NPZ

### 1) Материализованное представление с нормализованными полями

```sql
-- 1) безопасная переcборка
DROP MATERIALIZED VIEW IF EXISTS public.mv_candles_prepared;

CREATE MATERIALIZED VIEW public.mv_candles_prepared AS
SELECT
    k.symbol,
    -- unix ms -> UTC ts (timestamp without tz), по докам делим на 1000 и to_timestamp(...)
    -- затем фиксируем часовой пояс как UTC:
    (to_timestamp(k.open_time_ms / 1000.0) AT TIME ZONE 'UTC') AS ts_utc,
    k.open_price::double precision  AS open,
    k.high_price::double precision  AS high,
    k.low_price::double precision   AS low,
    k.close_price::double precision AS close,
    k.base_volume::double precision AS volume,
    CASE WHEN k.base_volume > 0
         THEN (k.quote_volume / k.base_volume)::double precision
         ELSE NULL
    END AS vwap,
    k.trade_count::integer AS trades
FROM public.klines_1m AS k
WHERE k.is_closed IS TRUE
WITH NO DATA;
```

Пояснения: конвертируем миллисекунды корректно (`to_timestamp(.../1000.0)`), приводим `numeric` → `double precision` (для numpy), берём только закрытые свечи, строим VWAP как `quote_volume/base_volume`. См. функции даты/времени в Postgres ([PostgreSQL][3]).

### 2) Индекс для конкурентного обновления MV

```sql
-- Unique индекс обязателен для REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS mv_candles_prepared_uq
ON public.mv_candles_prepared (symbol, ts_utc);
```

Это необходимо для `REFRESH MATERIALIZED VIEW CONCURRENTLY` (иначе будет блокирующий refresh) ([PostgreSQL][6]).

### 3) Первичное наполнение и последующие обновления

```sql
-- первичное наполнение (без CONCURRENTLY)
REFRESH MATERIALIZED VIEW public.mv_candles_prepared;

-- далее в кроне:
REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_candles_prepared;
```

([PostgreSQL][4])

> Альтернатива без MV: можно читать напрямую из `klines_1m`, рассчитывая поля «на лету». Но MV уменьшает нагрузку на CPU и упрощает консистентность выгрузок.

---

## Обновлённый SQL для выгрузки (если используем MV)

```sql
-- параметризованный селект под один символ/диапазон
SELECT symbol, ts_utc AS ts, open, high, low, close, volume, vwap, trades
FROM public.mv_candles_prepared
WHERE symbol = $1
  AND ts_utc >= $2
  AND ts_utc <  $3
ORDER BY ts_utc;
```

---

## Минимальный Python-скрипт выгрузки в NPZ (без хардкода, конфигами)

```python
# file: tools/export_to_npz.py
import os, numpy as np
import psycopg2
import json
from datetime import datetime, timezone

# читаем конфиг JSON/ENV, согласовано с README/SYSTEM_PROMPT (никакого хардкода путей/DSN)
# example JSON:
# {
#   "db": {"dsn": "host=... port=5432 dbname=... user=... password=..."},
#   "export": {"symbol": "BTCUSDT", "from_utc": "2024-01-01T00:00:00Z", "to_utc": "2025-01-01T00:00:00Z",
#              "out_path": "output/btcusdt_1m_2024.npz" }
# }

cfg = json.load(open(os.environ.get("EXPORT_CONFIG", "config/export_npz.json"), "r"))

q = """
SELECT ts_utc AS ts, open, high, low, close, volume, vwap, trades
FROM public.mv_candles_prepared
WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
ORDER BY ts_utc
"""

conn = psycopg2.connect(cfg["db"]["dsn"])
with conn, conn.cursor() as cur:
    cur.execute(q, (cfg["export"]["symbol"],
                    cfg["export"]["from_utc"],
                    cfg["export"]["to_utc"]))
    rows = cur.fetchall()

import numpy as np
if not rows:
    raise SystemExit("No rows in selected range")

# приводим к numpy. ts -> epoch ms (UTC) для удобства
ts = np.array([int(r[0].replace(tzinfo=timezone.utc).timestamp() * 1000) for r in rows], dtype=np.int64)
arr = {
    "ts": ts,
    "open":   np.array([r[1] for r in rows], dtype=np.float64),
    "high":   np.array([r[2] for r in rows], dtype=np.float64),
    "low":    np.array([r[3] for r in rows], dtype=np.float64),
    "close":  np.array([r[4] for r in rows], dtype=np.float64),
    "volume": np.array([r[5] for r in rows], dtype=np.float64),
    "vwap":   np.array([r[6] if r[6] is not None else np.nan for r in rows], dtype=np.float64),
    "trades": np.array([r[7] for r in rows], dtype=np.int32),
}
os.makedirs(os.path.dirname(cfg["export"]["out_path"]), exist_ok=True)
np.savez_compressed(cfg["export"]["out_path"], **arr)
print("Saved:", cfg["export"]["out_path"], "rows:", len(ts))
```

> Примечание: `to_timestamp` и `AT TIME ZONE` — поведение и типы описаны в оф. доках Postgres; будьте внимательны с `timestamp` vs `timestamptz` при дальнейших преобразованиях временных зон ([PostgreSQL][3]).

---

## Команды PowerShell (Windows) — как получить все нужные артефакты схемы

Замените плейсхолдеры. В PowerShell удобно задавать DSN через переменные окружения:

```powershell
# 1) Подготовка соединения
$env:PGHOST="localhost"; $env:PGPORT="5432"; $env:PGDATABASE="trading"; $env:PGUSER="postgres"; $env:PGPASSWORD="********"

# 2) Список схем / таблиц / представлений / индексов / прав / размеры
psql -v ON_ERROR_STOP=1 -c "\dn+"
psql -v ON_ERROR_STOP=1 -c "\dt+ public.*"
psql -v ON_ERROR_STOP=1 -c "\dv+ public.*"
psql -v ON_ERROR_STOP=1 -c "\di+ public.*"

# 3) Полное описание ключевых таблиц (колонки, типы, индексы)
psql -v ON_ERROR_STOP=1 -c "\d+ public.klines_1m"
psql -v ON_ERROR_STOP=1 -c "\d+ public.agg_trades"

# 4) Индексы аккуратно в CSV (удобно для diff)
psql -A -F "," -P footer=off -c "SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename, indexname" > .\indexes.csv

# 5) Размеры таблиц/матвью с аккуратным pretty-size
psql -A -F "," -P footer=off -c "SELECT relname, relkind, pg_size_pretty(pg_total_relation_size(oid)) AS total_size FROM pg_class WHERE relnamespace='public'::regnamespace AND relkind IN ('r','m') ORDER BY total_size DESC" > .\table_sizes.csv

# 6) Привилегии на объекты
psql -A -F "," -P footer=off -c "SELECT schemaname, tablename, privileges FROM information_schema.table_privileges WHERE schemaname='public' ORDER BY tablename, grantee" > .\table_privileges.csv
```

### Применение DDL (создание MV/индекса/refresh)

```powershell
# Пример: применяем DDL-файл
@'
DROP MATERIALIZED VIEW IF EXISTS public.mv_candles_prepared;
CREATE MATERIALIZED VIEW public.mv_candles_prepared AS
SELECT
  symbol,
  (to_timestamp(open_time_ms / 1000.0) AT TIME ZONE 'UTC') AS ts_utc,
  open_price::double precision  AS open,
  high_price::double precision  AS high,
  low_price::double precision   AS low,
  close_price::double precision AS close,
  base_volume::double precision AS volume,
  CASE WHEN base_volume > 0 THEN (quote_volume / base_volume)::double precision ELSE NULL END AS vwap,
  trade_count::integer AS trades
FROM public.klines_1m
WHERE is_closed IS TRUE
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS mv_candles_prepared_uq
ON public.mv_candles_prepared (symbol, ts_utc);
'@ | Out-File -Encoding UTF8 .\create_mv.sql

psql -v ON_ERROR_STOP=1 -f .\create_mv.sql
psql -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW public.mv_candles_prepared"
psql -v ON_ERROR_STOP=1 -c "REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_candles_prepared"
```

Документация по MV/refresh/индексации: ([PostgreSQL][4])

---

## Если хотите «правильно» закрепить уникальность на сырьевых таблицах

Рекомендация (опционально): формально оформить уникальность как constraint’ы — это улучшает семантику и помогает репликации/инструментам (вместо «только индекс») ([PostgreSQL][5]):

```sql
ALTER TABLE public.klines_1m
  ADD CONSTRAINT klines_1m_symbol_open_time_uk UNIQUE (symbol, open_time_ms);

ALTER TABLE public.agg_trades
  ADD CONSTRAINT agg_trades_symbol_agg_id_uk UNIQUE (symbol, agg_id);
```

---

## Шаг | Действие | KPI/риск

1 | Создать `mv_candles_prepared` и индекс | Быстрые выгрузки; риск: первый `REFRESH` блокирующий (разово)
2 | Перевести выгрузку NPZ на `mv_candles_prepared` | Стабильность формата; риск: забыть обновить MV по крону
3 | (Опционально) Добавить UNIQUE-констрейнты | Упрощение контроля качества; риск: конфликт дубликатов при миграции
4 | Проверка выгрузки (счётчик строк/минуты) | Целостность данных; риск: дыры в периодах
5 | Логи + мониторинг времени `REFRESH` | SLA ночных процедур; риск: рост времени при больших объёмах

---

### Почему именно так (ссылки на доки)

* Преобразование epoch-ms → timestamp: `to_timestamp(ms/1000.0)`; нюансы типов и TZ в оф. доках PostgreSQL ([PostgreSQL][3]).
* `AT TIME ZONE` — назначение/снятие TZ (для консистентного UTC) ([EDB][7]).
* Материализованные представления: создание/refresh, конкурентный refresh требует UNIQUE индекса ([PostgreSQL][4]).
* Отличия PK/UNIQUE и зачем формальные констрейнты, а не «просто индекс» ([PostgreSQL][5]).

---

## Готовность / приоритет

Готов к следующему шагу: **включить MV+экспорт в ваш пайплайн и (если требуется) оформить PR в репозитории под ветку `prosperous_bot`** с тест-гейтингом (`pytest`) и отчётом в `output/<config_name>/` согласно нашим правилам (см. SYSTEM_PROMPT/README) ([GitHub][2]). Если нужно — сразу подготовлю `unified diff` с командами `git/gh` под ваш путь к тулзам выгрузки.

Если хотите, могу так же адаптировать выгрузку без MV (селект прямо из `klines_1m`) — но MV даст более предсказуемое время и стабильность.

[1]: https://github.com/FMProducer/prosperous_bot/commit/931c2876836a47d016f514622b9d91870fbc721a "docs: tests websockets fix · FMProducer/prosperous_bot@931c287 · GitHub"
[2]: https://github.com/FMProducer/prosperous_bot/blob/prosperous_bot/third_party/rl-trading-binance/SYSTEM_PROMPT.md "prosperous_bot/third_party/rl-trading-binance/SYSTEM_PROMPT.md at prosperous_bot · FMProducer/prosperous_bot · GitHub"
[3]: https://www.postgresql.org/docs/current/functions-datetime.html?utm_source=chatgpt.com "Documentation: 18: 9.9. Date/Time Functions and Operators"
[4]: https://www.postgresql.org/docs/current/sql-creatematerializedview.html?utm_source=chatgpt.com "Documentation: 18: CREATE MATERIALIZED VIEW"
[5]: https://www.postgresql.org/docs/current/indexes-unique.html?utm_source=chatgpt.com "Documentation: 18: 11.6. Unique Indexes"
[6]: https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html?utm_source=chatgpt.com "Documentation: 18: REFRESH MATERIALIZED VIEW"
[7]: https://www.enterprisedb.com/postgres-tutorials/postgres-time-zone-explained?utm_source=chatgpt.com "Postgres AT TIME ZONE Explained"
