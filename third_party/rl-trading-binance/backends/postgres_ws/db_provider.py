import os
import pandas as pd
import psycopg2
from datetime import datetime, timezone
from typing import Optional, List, Iterator, Tuple

def get_feed(symbols: Optional[List[str]], start_utc: str, end_utc: str) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    DB Provider function to get klines data from PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "marketdata"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "9691"),
            connect_timeout=10
        )
    except psycopg2.OperationalError as e:
        raise RuntimeError(f"Could not connect to PostgreSQL database: {e}")

    start_ms = int(datetime.fromisoformat(start_utc.replace('Z', '+00:00')).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_utc.replace('Z', '+00:00')).timestamp() * 1000)

    if not symbols:
        # If no symbols are provided, get all symbols from the database within the time range.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM klines_1m WHERE open_time_ms >= %s AND open_time_ms <= %s",
                (start_ms, end_ms)
            )
            symbols = [row[0] for row in cur.fetchall()]

    for symbol in symbols:
        query = "SELECT * FROM klines_1m WHERE symbol = %s AND open_time_ms >= %s AND open_time_ms <= %s ORDER BY open_time_ms"
        
        try:
            df = pd.read_sql_query(query, conn, params=(symbol, start_ms, end_ms))
        except Exception as e:
            print(f"Error fetching data for symbol {symbol}: {e}")
            continue

        if df.empty:
            continue

        df['open_time_ms'] = pd.to_datetime(df['open_time_ms'], unit='ms', utc=True)
        df = df.set_index('open_time_ms')

        df.rename(columns={
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'base_volume': 'volume'
        }, inplace=True)

        yield (symbol, df)

    conn.close()
