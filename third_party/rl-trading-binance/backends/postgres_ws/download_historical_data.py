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
DATA_VISION_BASE = "https://data.binance.vision"
UM_MONTHLY_1M_PREFIX = "data/futures/um/monthly/klines"

TABLE = "public.klines_1m"

def month_range_last_complete(n_months: int):
    now = dt.datetime.now(dt.timezone.utc)
    first_this_month = dt.datetime(now.year, now.month, 1, tzinfo=dt.timezone.utc)
    last_prev_month = first_this_month - dt.timedelta(days=1)
    y, m = last_prev_month.year, last_prev_month.month
    out = []
    for _ in range(n_months):
        out.append((y, m))
        if m == 1:
            y -= 1
            m = 12
        else:
            m -= 1
    out.reverse()
    return out

def get_usdm_perp_symbols(session: requests.Session, timeout=20):
    r = session.get(EXCHANGE_INFO_URL, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    symbols = [s["symbol"] for s in data.get("symbols", []) if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"]
    return sorted(set(symbols))

def build_monthly_zip_url(symbol: str, year: int, month: int) -> str:
    filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    return f"{DATA_VISION_BASE}/{UM_MONTHLY_1M_PREFIX}/{symbol}/1m/{filename}"

def build_checksum_url(symbol: str, year: int, month: int) -> str:
    filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip.CHECKSUM"
    return f"{DATA_VISION_BASE}/{UM_MONTHLY_1M_PREFIX}/{symbol}/1m/{filename}"

def fetch_bytes(session: requests.Session, url: str, timeout=300):
    r = session.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content

def get_checksum_from_bytes(checksum_file_bytes: bytes) -> str | None:
    try:
        checksum_text = checksum_file_bytes.decode("utf-8", errors="ignore").strip()
        for part in checksum_text.replace("=", " ").split():
            if len(part) == 64 and all(c in "0123456789abcdefABCDEF" for c in part):
                return part.lower()
    except Exception:
        return None
    return None

def verify_checksum(content: bytes, remote_checksum: str) -> bool:
    if not remote_checksum:
        return False
    h = hashlib.sha256(content).hexdigest().lower()
    return h == remote_checksum

def setup_database():
    logger = logging.getLogger("um_klines_loader")
    conn = None
    try:
        conn = connect_pg()
        conn.autocommit = True
        with conn.cursor() as cur:
            logger.info("Проверка и создание таблицы...")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                symbol           text           NOT NULL,
                open_time_ms     bigint         NOT NULL,
                open_price       numeric(38,18) NOT NULL,
                high_price       numeric(38,18) NOT NULL,
                low_price        numeric(38,18) NOT NULL,
                close_price      numeric(38,18) NOT NULL,
                base_volume      numeric(38,18) NOT NULL,
                quote_volume     numeric(38,18) NOT NULL,
                trade_count      integer        NOT NULL,
                taker_base       numeric(38,18) NOT NULL,
                taker_quote      numeric(38,18) NOT NULL,
                is_closed        boolean        NOT NULL,
                ingest_ts        timestamptz    NOT NULL DEFAULT now(),
                monthly_checksum char(64)       DEFAULT NULL,
                PRIMARY KEY (symbol, open_time_ms)
            );
            """)
            logger.info("Проверка и создание индекса (может занять время при первом запуске)...")
            cur.execute(f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_klines_1m_checksum_check
            ON {TABLE} (symbol, date_trunc('month', to_timestamp(open_time_ms / 1000) AT TIME ZONE 'UTC'));
            """)
            logger.info("Инициализация таблицы завершена.")
    except psycopg2.Error as e:
        logger.error(f"Критическая ошибка при инициализации базы данных: {e}")
        if "CONCURRENTLY" in str(e):
            logger.warning("CREATE INDEX CONCURRENTLY не удался. Попытка создать обычный индекс...")
            conn_normal = None
            try:
                conn_normal = connect_pg()
                with conn_normal.cursor() as cur_normal:
                    cur_normal.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_klines_1m_checksum_check
                    ON {TABLE} (symbol, date_trunc('month', to_timestamp(open_time_ms / 1000) AT TIME ZONE 'UTC'));
                    """)
                conn_normal.commit()
                logger.info("Обычный индекс успешно создан.")
            except psycopg2.Error as e_inner:
                logger.error(f"Не удалось создать индекс и в обычном режиме: {e_inner}. Прерывание работы.")
                sys.exit(1)
            finally:
                if conn_normal:
                    conn_normal.close()
        else:
            sys.exit(1)
    finally:
        if conn:
            conn.close()

def get_existing_checksum(conn, symbol: str, year: int, month: int) -> str | None:
    start_ts = int(dt.datetime(year, month, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    if month == 12:
        end_ts = int(dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000) - 1
    else:
        end_ts = int(dt.datetime(year, month + 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000) - 1
    sql = f"SELECT DISTINCT monthly_checksum FROM {TABLE} WHERE symbol = %s AND open_time_ms >= %s AND open_time_ms <= %s AND monthly_checksum IS NOT NULL;"
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start_ts, end_ts))
        res = cur.fetchone()
    return res[0] if res else None

def parse_zip_klines(zip_bytes: bytes):
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return rows
        with zf.open(csv_names[0], "r") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for rec in reader:
                if len(rec) < 12:
                    continue
                try:
                    open_time_ms = int(rec[0])
                except ValueError:
                    continue
                rows.append((open_time_ms, rec[1], rec[2], rec[3], rec[4], rec[5], rec[7], int(rec[8]), rec[9], rec[10], True))
    return rows

def upsert_rows(conn, symbol: str, rows, checksum: str, batch_size=10_000):
    if not rows:
        return 0
    sql = f"""
    INSERT INTO {TABLE} (
        symbol, open_time_ms, open_price, high_price, low_price, close_price, base_volume,
        quote_volume, trade_count, taker_base, taker_quote, is_closed, monthly_checksum
    ) VALUES %s
    ON CONFLICT (symbol, open_time_ms) DO UPDATE SET
        open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price, low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price, base_volume = EXCLUDED.base_volume, quote_volume = EXCLUDED.quote_volume,
        trade_count = EXCLUDED.trade_count, taker_base = EXCLUDED.taker_base, taker_quote = EXCLUDED.taker_quote,
        is_closed = EXCLUDED.is_closed, ingest_ts = now(), monthly_checksum = EXCLUDED.monthly_checksum;
    """
    buf = [(symbol,) + r + (checksum,) for r in rows]
    with conn.cursor() as cur:
        execute_values(cur, sql, buf, page_size=batch_size)
    conn.commit()
    return len(buf)

def process_symbol_month(session, conn, symbol: str, year: int, month: int, do_verify_checksum: bool, logger: logging.Logger, timeout: int):
    cs_url = build_checksum_url(symbol, year, month)
    cs_bytes = fetch_bytes(session, cs_url, timeout=timeout)
    if cs_bytes is None:
        logger.info(f"[{symbol}] {year}-{month:02d}: 404 (нет CHECKSUM) — пропуск")
        return (symbol, year, month, 0, False)
    remote_checksum = get_checksum_from_bytes(cs_bytes)
    if not remote_checksum:
        logger.warning(f"[{symbol}] {year}-{month:02d}: не удалось распарсить CHECKSUM — пропуск")
        return (symbol, year, month, 0, False)
    if do_verify_checksum:
        local_checksum = get_existing_checksum(conn, symbol, year, month)
        if local_checksum and local_checksum == remote_checksum:
            logger.info(f"[{symbol}] {year}-{month:02d}: checksum совпал ({remote_checksum[:7]}...) — пропуск")
            return (symbol, year, month, 0, True)
    zip_url = build_monthly_zip_url(symbol, year, month)
    content = fetch_bytes(session, zip_url, timeout=timeout)
    if content is None:
        logger.info(f"[{symbol}] {year}-{month:02d}: 404 (нет архива) — пропуск")
        return (symbol, year, month, 0, False)
    if not verify_checksum(content, remote_checksum):
        logger.error(f"[{symbol}] {year}-{month:02d}: CHECKSUM НЕ СОВПАЛ — пропуск")
        return (symbol, year, month, 0, False)
    rows = parse_zip_klines(content)
    n = upsert_rows(conn, symbol, rows, remote_checksum)
    logger.info(f"[{symbol}] {year}-{month:02d}: загружено/обновлено {n} строк с checksum {remote_checksum[:7]}...")
    return (symbol, year, month, n, True)

def connect_pg():
    conn = psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "marketdata"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        connect_timeout=10
    )
    conn.autocommit = False
    return conn

def main():
    parser = argparse.ArgumentParser(description="Загрузчик месячных klines с Binance Vision в PostgreSQL.")
    parser.add_argument("--months", type=int, default=3, help="Сколько полных месяцев загрузить.")
    parser.add_argument("--workers", type=int, default=8, help="Количество потоков.")
    parser.add_argument("--verify-checksum", action="store_true", help="Проверять checksum перед загрузкой.")
    parser.add_argument("--symbols", type=str, default="", help="Фильтр символов через запятую.")
    parser.add_argument("--timeout", type=int, default=300, help="Тайм-аут для HTTP запросов.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("um_klines_loader")

    months = month_range_last_complete(args.months)
    logger.info(f"Будут загружены месяцы: {months}")

    setup_database()

    with requests.Session() as http:
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
        def worker(symbol, ym):
            y, m = ym
            with connect_pg() as local_conn:
                return process_symbol_month(http, local_conn, symbol, y, m, do_verify_checksum=args.verify_checksum, logger=logger, timeout=args.timeout)

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
