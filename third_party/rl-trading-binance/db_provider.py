import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from typing import Optional, List, Iterator, Tuple

def get_feed(symbols: Optional[List[str]], start_utc: str, end_utc: str) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    DB Provider function to get klines data from PostgreSQL database using SQLAlchemy.
    """
    try:
        password = os.getenv('PGPASSWORD')
        if not password:
            raise RuntimeError("PGPASSWORD environment variable not set.")
        db_url = f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:{password}@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')}/{os.getenv('PGDATABASE', 'marketdata')}"
        engine = create_engine(db_url, connect_args={'connect_timeout': 10})
    except Exception as e:
        raise RuntimeError(f"Could not create SQLAlchemy engine: {e}")

    start_ms = int(datetime.fromisoformat(start_utc.replace('Z', '+00:00')).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_utc.replace('Z', '+00:00')).timestamp() * 1000)

    if not symbols:
        # If no symbols are provided, get all symbols from the database within the time range.
        try:
            with engine.connect() as connection:
                result = connection.execute(
                    text("SELECT DISTINCT symbol FROM klines_1m WHERE open_time_ms >= :start_ms AND open_time_ms <= :end_ms"),
                    {'start_ms': start_ms, 'end_ms': end_ms}
                )
                symbols = [row[0] for row in result]
        except Exception as e:
            raise RuntimeError(f"Could not fetch symbol list from database: {e}")


    # Определим доступные колонки в таблице (один раз на соединение)
    try:
        with engine.connect() as connection:
            cols_res = connection.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'klines_1m'")
            )
            available_cols = {row[0] for row in cols_res}
    except Exception as e:
        raise RuntimeError(f"Could not inspect table columns: {e}")

    base_cols = ["open_time_ms", "open_price", "high_price", "low_price", "close_price", "base_volume"]
    opt_cols = []
    # Добавим только реально существующие «опциональные» поля
    if "quote_asset_volume" in available_cols:
        opt_cols.append("quote_asset_volume")
    # num_trades не обязателен — используем только если есть
    if "num_trades" in available_cols:
        opt_cols.append("num_trades")

    select_cols = base_cols + opt_cols
    select_clause = ", ".join(select_cols)

    for symbol in symbols:
        query = (
            f"SELECT {select_clause} FROM klines_1m "
            "WHERE symbol = :symbol AND open_time_ms >= :start_ms AND open_time_ms <= :end_ms "
            "ORDER BY open_time_ms"
        )
        
        try:
            df = pd.read_sql_query(sql=text(query), con=engine, params={'symbol': symbol, 'start_ms': start_ms, 'end_ms': end_ms})
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
        # Рассчитываем VWAP-подобную метрику, только если есть quote_asset_volume
        if 'quote_asset_volume' in df.columns:
            df['volume_weighted_average'] = df['quote_asset_volume'] / (df['volume'] + 1e-9)
            df.drop(columns=['quote_asset_volume'], inplace=True)

        yield (symbol, df)
