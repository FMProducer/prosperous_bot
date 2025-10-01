#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скачивает последние N (по умолчанию 3) завершённых месяцев monthly 1m klines
для всех USDⓈ-M PERPETUAL символов Binance и пишет в Postgres (upsert).

Требуемые переменные окружения для подключения к БД:
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Примеры запуска:
  python download_historical_data.py --months 3 --workers 8 --verify-checksum

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
        symbol        text        NOT NULL,
        open_time_ms  bigint      NOT NULL, -- k.t (ms)
        open_price    numeric(38,18) NOT NULL,
        high_price    numeric(38,18) NOT NULL,
        low_price     numeric(38,18) NOT NULL,
        close_price   numeric(38,18) NOT NULL,
        base_volume   numeric(38,18) NOT NULL, -- k.v
        quote_volume  numeric(38,18) NOT NULL, -- k.q
        trade_count   integer     NOT NULL,    -- k.n
        taker_base    numeric(38,18) NOT NULL, -- k.V
        taker_quote   numeric(38,18) NOT NULL, -- k.Q
        is_closed     boolean     NOT NULL,    -- k.x
        ingest_ts     timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (symbol, open_time_ms)
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()

def parse_zip_klines(zip_bytes: bytes):
    """Возвращает список строк для вставки: (open_time,...,is_closed) в правильных типах."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Берём первый .csv внутри (в monthly он один)
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return rows
        with zf.open(csv_names[0], "r") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for rec in reader:
                if len(rec) < 12:
                    continue
                # Приводим типы
                open_time_ms = int(rec[0])
                open_price = rec[1]
                high_price = rec[2]
                low_price = rec[3]
                close_price = rec[4]
                base_volume = rec[5]
                # close_time = int(rec[6]) # не используется в целевой схеме
                quote_volume = rec[7]
                trade_count = int(rec[8])
                taker_base = rec[9]
                taker_quote = rec[10]
                # ignore = rec[11] # не используется в целевой схеме
                is_closed = True # Исторические данные всегда закрыты

                rows.append((
                    open_time_ms, open_price, high_price, low_price, close_price, base_volume,
                    quote_volume, trade_count, taker_base, taker_quote, is_closed
                ))
    return rows

def upsert_rows(conn, symbol: str, rows, batch_size=10_000):
    if not rows:
        return 0
    inserted = 0
    # 12 колонок: symbol + 11 из parse_zip_klines
    tpl = "(" + ",".join(["%s"] * 12) + ")"
    sql = f"""
    INSERT INTO {TABLE} (
        symbol, open_time_ms, open_price, high_price, low_price, close_price, base_volume,
        quote_volume, trade_count, taker_base, taker_quote, is_closed
    ) VALUES %s
    ON CONFLICT (symbol, open_time_ms) DO UPDATE SET
        open_price = EXCLUDED.open_price,
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        base_volume = EXCLUDED.base_volume,
        quote_volume = EXCLUDED.quote_volume,
        trade_count = EXCLUDED.trade_count,
        taker_base = EXCLUDED.taker_base,
        taker_quote = EXCLUDED.taker_quote,
        is_closed = EXCLUDED.is_closed,
        ingest_ts = now()
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
