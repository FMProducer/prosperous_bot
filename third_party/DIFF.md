---

# TL;DR

Сейчас ровно по 525 600 минутных баров на тикер за год (полный непрерывный год). Для быстрых выгрузок непрерывных периодов в `.npz` и долгосрочных бэктестов:

1. **Оставить текущую таблицу, но навести «порядок под выгрузку»:**
   — удалить дублирующий уникальный индекс (у вас два ~10 GB),
   — добавить компактный **BRIN** по времени для ускорения range-запросов,
   — сделать **view**/материализованное представление с нужными колонками и корректным **VWAP**,
   — завести **coverage-MV** с min/max/count по тикеру (для быстрых проверок охвата).
   Это даст мгновенные выгрузки и экономит место/IO. BRIN официально рекомендован для «почти отсортированных по времени» очень больших таблиц. ([PostgreSQL][2])

2. **Экспортер в NPZ** (готовый шаблон Python): вытягивает минутные окна `[start; end)` из view, складывает `ts` (мс) + каналы **строго** в порядке `DataConfig`. Используется `COPY`/`\copy` как самый быстрый способ выгрузки запроса из Postgres. ([PostgreSQL][3])

---

## Шаг | Действие | KPI/риск

1 | Проверить и удалить лишний уникальный индекс | −10 GB на диск, быстрее `INSERT/UPDATE`; риск: удалить не тот индекс → **проверка DDL обяз.**
2 | Создать BRIN по времени (и по разделам, если будут) | Быстрые range-сканы, маленький индекс; риск: слабая корреляция по времени → уменьшить `pages_per_range`. ([PostgreSQL][2])
3 | Создать `VIEW` с колонками {`ts`,`open`,`high`,`volume_weighted_average`,`low`,`close`,`volume`,`num_trades`} | Гарантия схемы под `DataConfig`; риск: неверный тип времени → см. авто-детект ниже. 
4 | Материализованное `coverage`-представление + `REFRESH CONCURRENTLY` | Быстрая оценка охвата данных; риск: нужен уникальный индекс для `CONCURRENTLY`. ([PostgreSQL][5])
5 | Экспортер NPZ (скрипт) + шаблоны `psql \copy` | Скорость выгрузки; риск: несоответствие порядку каналов → жёстко фиксируем список из `DataConfig`. ([PostgreSQL][3])
6 | (Опц.) Партиционирование / Timescale + компрессия | Масштабируемость и экономия места; риск: миграция данных потребует времени. ([PostgreSQL][4])

---

## 1) Индексы: ревизия и BRIN

**Анализ ваших индексов:** два индекса по ~10 GB (`klines_1m_pkey` и `klines_1m_symbol_open_time_uk`) выглядят дубликатами по ключу (`symbol, open_time`). Сначала проверяем DDL:

```sql
-- Посмотреть определение индексов
SELECT i.relname AS index_name, pg_get_indexdef(ix.indexrelid) AS indexdef
FROM pg_index ix
JOIN pg_class t ON t.oid = ix.indrelid
JOIN pg_class i ON i.oid = ix.indexrelid
WHERE t.relname = 'klines_1m'
ORDER BY pg_relation_size(ix.indexrelid) DESC;
```

Если `klines_1m_symbol_open_time_uk` дублирует PK, удалить **только** его:

```sql
DROP INDEX IF EXISTS public.klines_1m_symbol_open_time_uk;
```

**BRIN по времени** (очень маленький и быстрый на time-range):

```sql
-- если колонка времени называется open_time (timestamptz)
CREATE INDEX IF NOT EXISTS klines_1m_brin_open_time
ON public.klines_1m
USING brin (open_time) WITH (pages_per_range = 128);
-- При слабой корреляции снизьте до 32/64. :contentReference[oaicite:11]{index=11}
```

> Почему BRIN: компактный индекс по блокам, эффективен для «естественно отсортированных» временных колонок и строится одной линейной прогонкой по таблице. ([PostgreSQL][2])

---

## 2) Представление под NPZ (+ корректный VWAP)

Формат **должен совпадать** с `DataConfig.expected_channels` (см. `config.py`), иначе селектор каналов в бэктестах собьётся. 

```sql
-- Авто-детект формата времени и сборка нужных колонок:
-- 1) Узнаём имя и тип временного поля
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema='public' AND table_name='klines_1m'
  AND column_name ~* '(open_)?time|ts';

-- 2) Создаём VIEW с правильным ts (мс)
CREATE OR REPLACE VIEW public.v_klines_1m_npz AS
SELECT
  symbol,
  -- ВАРИАНТ A: если open_time = timestamptz
  (EXTRACT(EPOCH FROM open_time)::bigint * 1000)            AS ts,
  -- ВАРИАНТ B: если open_time = bigint (мс) — тогда просто: open_time AS ts,
  open::float8                                              AS open,
  high::float8                                              AS high,
  CASE WHEN volume > 0 THEN quote_asset_volume / volume END AS volume_weighted_average,
  low::float8                                               AS low,
  close::float8                                             AS close,
  volume::float8                                            AS volume,
  number_of_trades::float8                                  AS num_trades
FROM public.klines_1m;
```

> Если хотите «зацементировать» VWAP, можно добавить STORED-колонку и поддерживать её триггером, но `VIEW` проще и без накладных расходов на запись.

---

## 3) Coverage-MV (быстрый контроль полноты диапазонов)

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_klines_coverage AS
SELECT
  symbol,
  MIN(open_time) AS first_ts,
  MAX(open_time) AS last_ts,
  COUNT(*)       AS rows_cnt
FROM public.klines_1m
GROUP BY symbol;

-- Для REFRESH CONCURRENTLY нужен уникальный индекс:
CREATE UNIQUE INDEX IF NOT EXISTS mv_klines_coverage_symbol_uidx
  ON public.mv_klines_coverage (symbol);

-- Обновление без блокировки SELECT'ов:
REFRESH MATERIALIZED VIEW CONCURRENTLY public.mv_klines_coverage;
-- CONCURRENTLY допускается с уникальным индексом. :contentReference[oaicite:14]{index=14}
```

---

## 4) Экспорт непрерывного окна в NPZ (готовый скрипт)

Ниже — минимальный экспортер. Он:

* принимает `--start-utc`, `--end-utc`, `--symbols` (через запятую или `ALL`),
* тянет из `v_klines_1m_npz`, сортирует,
* сохраняет NPZ **в канальном порядке из `DataConfig`** (обязательно),
* совместим с `backtest_continuous.py`, где ожидается `ts` в мс и `df[cfg.data.data_channels]`. 

```python
# tools/export_to_npz.py
import argparse, numpy as np, pandas as pd, psycopg2, os
from psycopg2.extras import RealDictCursor

CHANNELS = ["open","high","volume_weighted_average","low","close","volume","num_trades"]  # must match DataConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help="postgresql://user:pass@host:5432/marketdata")
    ap.add_argument("--start-utc", required=True)   # "2024-10-01 00:00:00"
    ap.add_argument("--end-utc",   required=True)   # "2025-10-01 00:00:00"
    ap.add_argument("--symbols",   default="ALL")   # "BTCUSDT,ETHUSDT" or ALL
    ap.add_argument("--out",       default="data/backtest_data.npz")
    args = ap.parse_args()

    where = ["ts >= EXTRACT(EPOCH FROM %s)::bigint*1000",
             "ts <  EXTRACT(EPOCH FROM %s)::bigint*1000"]
    params = [args.start_utc, args.end_utc]

    if args.symbols != "ALL":
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        where.append("symbol = ANY(%s)")
        params.append(syms)

    sql = f"""
        WITH base AS (
            SELECT symbol, ts, {", ".join(CHANNELS)}
            FROM v_klines_1m_npz
            WHERE {" AND ".join(where)}
        )
        SELECT ts, {", ".join(CHANNELS)}
        FROM base
        ORDER BY ts, symbol;
    """

    with psycopg2.connect(args.dsn) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        raise SystemExit("No rows in requested interval")

    df = pd.DataFrame(rows)
    # Сохраняем ключами, которые потом читает backtest_continuous.py
    np.savez_compressed(
        args.out,
        ts=df["ts"].astype("int64").values,
        **{ch: df[ch].astype("float32").values for ch in CHANNELS}
    )
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
```

**PowerShell пример запуска:**

```powershell
$DSN = "postgresql://postgres:9691@localhost:5432/marketdata"
python tools/export_to_npz.py --dsn $DSN `
  --start-utc "2025-01-01 00:00:00" `
  --end-utc   "2025-03-01 00:00:00" `
  --symbols   "BTCUSDT,ETHUSDT" `
  --out "data/backtest_data_2025Q1_BTC_ETH.npz"
```

> Почему так быстро: вытягиваем ровно нужное окно; при необходимости можно заменить `SELECT` на `\copy (SELECT …) TO STDOUT` для потоковой выгрузки (самый быстрый путь экспорта SQL-результата из Postgres). ([PostgreSQL][3])

---

## 5) Проверка непрерывности минут (по тикеру и окну)

```sql
-- Проверка «дыр» за период по одному тикеру
WITH grid AS (
  SELECT generate_series(
    TIMESTAMP '2025-01-01 00:00:00',
    TIMESTAMP '2025-03-01 00:00:00' - INTERVAL '1 minute',
    INTERVAL '1 minute'
  ) AS ts
)
SELECT g.ts
FROM grid g
LEFT JOIN public.klines_1m k
  ON k.symbol = 'BTCUSDT'
 AND k.open_time = g.ts
WHERE k.open_time IS NULL
ORDER BY g.ts
LIMIT 50;
```

---

## 6) (Опционально) Долгосрочная масштабируемость

### Вариант A — Партиционирование PostgreSQL

Партиционирование по месяцу (RANGE по времени) и наследование индексов у разделов. Хорошо ложится на time-range выгрузки и облегчает `VACUUM/REINDEX`. ([PostgreSQL][4])

```sql
-- Пример новой таблицы с партиционированием (эскиз)
CREATE TABLE public.klines_1m_v2 (
  symbol text NOT NULL,
  open_time timestamptz NOT NULL,
  open double precision,
  high double precision,
  low  double precision,
  close double precision,
  volume double precision,
  quote_asset_volume double precision,
  number_of_trades bigint,
  PRIMARY KEY(symbol, open_time)
) PARTITION BY RANGE (open_time);

-- Раздел за январь-2025
CREATE TABLE public.klines_1m_2025_01 PARTITION OF public.klines_1m_v2
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
-- Индексы на разделах + BRIN по open_time:
CREATE INDEX klines_1m_2025_01_brin ON public.klines_1m_2025_01 USING brin(open_time);
```

### Вариант B — TimescaleDB Hypertable + компрессия

Если можно подключить расширение — перевод таблицы в **hypertable** (`create_hypertable`) с чанками по времени (и доп.измерением `symbol`), плюс **compression policy** для старых чанков. Это даст авто-менеджмент чанков и ощутимую экономию места. ([docs.tigerdata.com][6])

---

## 7) Как это вяжется с вашим бэктестом

* `backtest_continuous.py` грузит NPZ и ожидает индекс `ts` (мс) и матрицу признаков **в порядке `cfg.data.data_channels`**. Мы именно так сохраняем массивы; значит сравнение «сегментированное vs непрерывное» будет корректным. 
* Параметры окна агента (`full_seq_len=150`, `pre_signal_len=90`, `post_signal_len=60`) берутся из `config.py` и остаются неизменными — непрерывные данные просто «скользящим окном» режутся внутри `backtest_continuous.py`.

---

### Нужные уточнения (минимум)

Чтобы сразу накатить SQL без правок, ответьте, пожалуйста, одним сообщением:

1. точное имя и тип колонки времени в `public.klines_1m` (например, `open_time timestamptz` или `open_time bigint(ms)`),
2. точные имена колонок для `quote_asset_volume` и `number_of_trades` (они нужны для корректного VWAP и `num_trades`).

После этого пришлю готовый блок PowerShell/`psql`-команд для вашего окружения Windows, плюс проверочные запросы и шаблоны для разных периодов.

---

**Напоминание по правилам репо:** каналы/порядок — строго по `DataConfig`, результаты/логи — в `output/<config_name>/`. 
