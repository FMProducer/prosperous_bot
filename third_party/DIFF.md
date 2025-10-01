# TL;DR

Ниже — готовый Python-скрипт, который:

1. берёт **актуальный список USDⓈ-M perpetual символов** с Binance через `GET /fapi/v1/exchangeInfo`, фильтруя `status=TRADING` и `contractType=PERPETUAL`. ([Бинанс Разработчик][2])
2. скачивает **monthly 1m klines** с **data.binance.vision** за **последние 3 полностью завершённых месяца** по всем символам: путь `data/futures/um/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-YYYY-MM.zip` (+ опциональная проверка `.CHECKSUM`). ([data.binance.vision][3])
3. парсит CSV из ZIP и **загружает в вашу PostgreSQL** (таблица `public.klines_1m`) с upsert по `(symbol, open_time)`.

> Отдельный .npz-экспорт для бэктестера у вас уже есть, поэтому в этом скрипте его **не делаем** (соответствует вашим требованиям и проектной доктрине: конфиги/артефакты — разнесены, секреты — только через переменные окружения). 

---

| Шаг | Действие                                                                    | KPI / риск                                                                                                                                                                                         |
| --- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Получение вселенной USDⓈ-M PERPETUAL из `/fapi/v1/exchangeInfo`             | Быстро, 1 вес/запрос; риск — временные лимиты API. ([Бинанс Разработчик][2])                                                                                                                       |
| 2   | Скачивание monthly `1m` ZIP с `data.binance.vision` за 3 завершённых месяца | Высокая скорость (архивы по месяцу); риск — отсутствующий архив за текущий месяц (мы берём **полные** месяцы). Обновление monthly — на первой неделе следующего месяца. ([data.binance.vision][3]) |
| 3   | Вставка в Postgres с upsert, PK `(symbol, open_time)`                       | Идempotentность, отсутствие дублей; риск — размеры batch’ей (настроены). Политика секретов — через env.                                                                                            |

---

# Скрипт: `download_um_1m_to_postgres.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скачивает последние N (по умолчанию 3) завершённых месяцев monthly 1m klines
для всех USDⓈ-M PERPETUAL символов Binance и пишет в Postgres (upsert).

Требуемые переменные окружения для подключения к БД:
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Примеры запуска:
  python download_um_1m_to_postgres.py --months 3 --workers 8 --verify-checksum

Документация и источники:
- exchangeInfo (Futures USDⓈ-M): GET /fapi/v1/exchangeInfo
- data.binance.vision: data/futures/um/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-YYYY-MM.zip
- .CHECKSUM файлы лежат рядом с архивами

"""

import os
import io
import csv
import sys
import time
import math
import json
import gzip
import argparse
import hashlib
import zipfile
import logging
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import psycopg2
from psycopg2.extras import execute_values

BINANCE_FAPI_BASE = "https://fapi.binance.com"
EXCHANGE_INFO_URL = f"{BINANCE_FAPI_BASE}/fapi/v1/exchangeInfo"
# Пример пути: data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-07.zip
DATA_VISION_BASE = "https://data.binance.vision"
UM_MONTHLY_1M_PREFIX = "data/futures/um/monthly/klines"

# Таблица назначения
TABLE = "public.klines_1m"

# Колонки csv (см. Binance klines формат)
# Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades,
# Taker buy base asset volume, Taker buy quote asset volume, Ignore
# (источник формата колонок — Binance public data README + Klines docs)
CSV_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
]

def month_range_last_complete(n_months: int):
    """Вернуть список (year, month) для n последних полностью завершённых месяцев, начиная с прошлого месяца."""
    now = dt.datetime.utcnow()
    # Переходим к первому числу текущего месяца и затем -1 день -> последний день прошлого месяца
    first_this_month = dt.datetime(now.year, now.month, 1)
    last_prev_month = first_this_month - dt.timedelta(days=1)
    y, m = last_prev_month.year, last_prev_month.month
    out = []
    for _ in range(n_months):
        out.append((y, m))
        # шаг назад на 1 месяц
        if m == 1:
            y -= 1
            m = 12
        else:
            m -= 1
    out.reverse()
    return out

def get_usdm_perp_symbols(session: requests.Session, timeout=20):
    """Забрать все USDT-M perpetual symbols со статусом TRADING."""
    r = session.get(EXCHANGE_INFO_URL, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    symbols = []
    for s in data.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL":
            # На USDⓈ-M фьючерсах baseAsset/quoteAsset обычно USDT кроссы;
            # используем символ как есть (верхний регистр нужен для путей на data.binance.vision).
            symbols.append(s["symbol"])
    symbols = sorted(set(symbols))
    return symbols

def build_monthly_zip_url(symbol: str, year: int, month: int) -> str:
    filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    return f"{DATA_VISION_BASE}/{UM_MONTHLY_1M_PREFIX}/{symbol}/1m/{filename}"

def build_checksum_url(symbol: str, year: int, month: int) -> str:
    filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip.CHECKSUM"
    return f"{DATA_VISION_BASE}/{UM_MONTHLY_1M_PREFIX}/{symbol}/1m/{filename}"

def fetch_bytes(session: requests.Session, url: str, timeout=60):
    r = session.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content

def verify_checksum(content: bytes, checksum_file_bytes: bytes) -> bool:
    """Проверка sha256 по формату .CHECKSUM (строка 'SHA256 (<file>) = <hash>' или просто '<hash>  <file>')."""
    try:
        checksum_text = checksum_file_bytes.decode("utf-8", errors="ignore").strip()
        # Извлекаем hex-строку sha256
        token = None
        for part in checksum_text.replace("=", " ").split():
            if len(part) == 64 and all(c in "0123456789abcdefABCDEF" for c in part):
                token = part.lower()
                break
        if not token:
            return False
        h = hashlib.sha256(content).hexdigest().lower()
        return h == token
    except Exception:
        return False

def ensure_table(conn):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        symbol TEXT NOT NULL,
        open_time BIGINT NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume NUMERIC,
        close_time BIGINT,
        quote_asset_volume NUMERIC,
        number_of_trades INTEGER,
        taker_buy_base_asset_volume NUMERIC,
        taker_buy_quote_asset_volume NUMERIC,
        ignore NUMERIC,
        CONSTRAINT klines_1m_pk PRIMARY KEY (symbol, open_time)
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()

def parse_zip_klines(zip_bytes: bytes):
    """Возвращает список строк для вставки: (open_time,...,ignore) в правильных типах."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Берём первый .csv внутри (в monthly он один)
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return rows
        with zf.open(csv_names[0], "r") as f:
            for raw in io.TextIOWrapper(f, encoding="utf-8", newline=""):
                # Быстрый csv-парсер (на случай разделителей внутри — лучше использовать csv.reader)
                # Здесь используем csv.reader явно:
                pass
    # Перечитаем через csv.reader корректно
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        with zf.open(csv_names[0], "r") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for rec in reader:
                if len(rec) < 12:
                    continue
                # Приводим типы
                open_time = int(rec[0])
                open_ = rec[1]
                high = rec[2]
                low = rec[3]
                close = rec[4]
                volume = rec[5]
                close_time = int(rec[6])
                quote_asset_volume = rec[7]
                number_of_trades = int(rec[8])
                taker_buy_base = rec[9]
                taker_buy_quote = rec[10]
                ignore = rec[11]
                rows.append((
                    open_time, open_, high, low, close, volume, close_time,
                    quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote, ignore
                ))
    return rows

def upsert_rows(conn, symbol: str, rows, batch_size=10_000):
    if not rows:
        return 0
    inserted = 0
    tpl = "(" + ",".join(["%s"] * 13) + ")"
    sql = f"""
    INSERT INTO {TABLE} (
        symbol, open_time, open, high, low, close, volume, close_time,
        quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume, ignore
    ) VALUES %s
    ON CONFLICT (symbol, open_time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        close_time = EXCLUDED.close_time,
        quote_asset_volume = EXCLUDED.quote_asset_volume,
        number_of_trades = EXCLUDED.number_of_trades,
        taker_buy_base_asset_volume = EXCLUDED.taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume = EXCLUDED.taker_buy_quote_asset_volume,
        ignore = EXCLUDED.ignore
    """
    buf = []
    for r in rows:
        buf.append(
            (symbol,) + r  # prepend symbol
        )
        if len(buf) >= batch_size:
            with conn.cursor() as cur:
                execute_values(cur, sql, buf, page_size=10000)
            conn.commit()
            inserted += len(buf)
            buf.clear()
    if buf:
        with conn.cursor() as cur:
            execute_values(cur, sql, buf, page_size=10000)
        conn.commit()
        inserted += len(buf)
    return inserted

def process_symbol_month(session, conn, symbol: str, year: int, month: int, verify_checksum: bool, logger: logging.Logger):
    zip_url = build_monthly_zip_url(symbol, year, month)
    content = fetch_bytes(session, zip_url)
    if content is None:
        logger.info(f"[{symbol}] {year}-{month:02d}: 404 (нет архива) — пропуск")
        return (symbol, year, month, 0, False)
    if verify_checksum:
        cs_url = build_checksum_url(symbol, year, month)
        cs_bytes = fetch_bytes(session, cs_url)
        if cs_bytes is None:
            logger.warning(f"[{symbol}] {year}-{month:02d}: отсутствует CHECKSUM — продолжаем без проверки")
        else:
            ok = verify_checksum(content, cs_bytes)
            if not ok:
                logger.error(f"[{symbol}] {year}-{month:02d}: CHECKSUM НЕ СОВПАЛ — пропуск архива")
                return (symbol, year, month, 0, False)
    rows = parse_zip_klines(content)
    n = upsert_rows(conn, symbol, rows)
    logger.info(f"[{symbol}] {year}-{month:02d}: загружено {n} строк")
    return (symbol, year, month, n, True)

def connect_pg():
    conn = psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "marketdata"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "")
    )
    conn.autocommit = False
    return conn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=3, help="Сколько полных месяцев загрузить (по умолчанию 3).")
    parser.add_argument("--workers", type=int, default=8, help="Количество потоков для скачивания.")
    parser.add_argument("--verify-checksum", action="store_true", help="Проверять .CHECKSUM для архивов.")
    parser.add_argument("--symbols", type=str, default="", help="Кому-сепарированный фильтр символов (опц.).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("um_klines_loader")

    months = month_range_last_complete(args.months)
    logger.info(f"Будут загружены месяцы: {months}")

    with requests.Session() as http, connect_pg() as conn:
        ensure_table(conn)

        all_symbols = get_usdm_perp_symbols(http)
        logger.info(f"Всего активных USDⓈ-M PERPETUAL символов: {len(all_symbols)}")

        if args.symbols:
            filter_set = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
            symbols = [s for s in all_symbols if s in filter_set]
            logger.info(f"Фильтр по символам: {len(symbols)} из {len(all_symbols)}")
        else:
            symbols = all_symbols

        tasks = []
        results = []
        # ВАЖНО: один pg-conn не потокобезопасен; создадим соединения по месту выполнения
        def worker(symbol, ym):
            y, m = ym
            with connect_pg() as local_conn:
                return process_symbol_month(http, local_conn, symbol, y, m, args.verify_checksum, logger)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for sym in symbols:
                for ym in months:
                    tasks.append(ex.submit(worker, sym, ym))
            for fut in as_completed(tasks):
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.exception("Ошибка выполнения задачи: %s", e)

        total = sum(r[3] for r in results if r and r[4])
        ok = sum(1 for r in results if r and r[4] and r[3] > 0)
        logger.info(f"Готово. Успешных загрузок: {ok}, всего вставлено строк: {total}.")

if __name__ == "__main__":
    main()
```

## Зависимости

```bash
pip install requests psycopg2-binary
```

## Настройки окружения

```bash
# Пример (Windows PowerShell)
$env:PGHOST="localhost"
$env:PGPORT="5432"
$env:PGDATABASE="marketdata"
$env:PGUSER="postgres"
$env:PGPASSWORD="<Ваш_пароль>"
```

## DDL (если таблицы ещё нет)

Скрипт сам создаёт `public.klines_1m`, но на всякий случай — DDL:

```sql
CREATE TABLE IF NOT EXISTS public.klines_1m (
  symbol TEXT NOT NULL,
  open_time BIGINT NOT NULL,
  open NUMERIC,
  high NUMERIC,
  low NUMERIC,
  close NUMERIC,
  volume NUMERIC,
  close_time BIGINT,
  quote_asset_volume NUMERIC,
  number_of_trades INTEGER,
  taker_buy_base_asset_volume NUMERIC,
  taker_buy_quote_asset_volume NUMERIC,
  ignore NUMERIC,
  CONSTRAINT klines_1m_pk PRIMARY KEY (symbol, open_time)
);
```

---

## Почему именно так (ссылки на первоисточники)

* `GET /fapi/v1/exchangeInfo` — официальный способ получить текущие правила/символы для USDⓈ-M Futures. Мы фильтруем `TRADING` + `PERPETUAL`. ([Бинанс Разработчик][2])
* Monthly 1m klines на **data.binance.vision**: каталог `data/futures/um/monthly/klines/…` (индекс-браузер), загружаем по **всем символам**, присутствующим в каталоге. ([data.binance.vision][3])
* Формат колонок Klines (12 полей) соответствует документации/README Binance public data. ([Бинанс Разработчик][4])
* Для проверки целостности из каталога берём `.zip.CHECKSUM` рядом с архивом (опция `--verify-checksum`). ([GitHub][5])
* Monthly-архивы пополняются **в начале следующего месяца** (официальный public-data README). Поэтому мы качаем **последние N полностью завершённых месяцев** — без риска частично заполненного текущего. ([GitHub][5])

---

## Соответствие процессу RL Trader

* Скрипт не трогает .npz-экспорт и бэктестер (у вас уже есть отдельный шаг/скрипт), — следуем доктрине «конфиги и артефакты — разнесены; секреты — через env». 
* Потоки/батчи сделаны для скорости, но с **идемпотентным upsert** по `(symbol, open_time)` — безопасно для последующих повторных прогонов.
* В проектной документации RL Trader подчёркнуто, что продакшн-пайплайн использует **real-time WebSocket + DB**, а бэктест — на статике (.npz). Наш скрипт как раз наполняет БД историей, что согласуется с общей архитектурой. 

---

Если хотите, в следующем шаге добавлю **unit-тесты** (offline, с локальными фикстурами ZIP/CSV) и **CLI-обвязку в ваш репозиторий** под `third_party/rl-trading-binance/tools/`, оформив PR по нашему шаблону. 

[1]: https://github.com/FMProducer/prosperous_bot/commit/196d89c360ff3623c45e0c108ee95e2194fbfd7e "docs: export_npz · FMProducer/prosperous_bot@196d89c · GitHub"
[2]: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information?utm_source=chatgpt.com "Exchange Information | Binance Open Platform"
[3]: https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fmonthly%2Fklines%2F&utm_source=chatgpt.com "Home / data / futures / um / monthly / klines"
[4]: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data?utm_source=chatgpt.com "Kline Candlestick Data | Binance Open Platform"
[5]: https://github.com/binance/binance-public-data?utm_source=chatgpt.com "Details on how to get Binance public data"
