import datetime as dt

import logging

import os

import sys

import importlib.util

from typing import List, Tuple

import numpy as np

import pandas as pd

from sqlalchemy import create_engine, text

from tqdm import tqdm

from config import MasterConfig

from utils import load_config

def setup_logging():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s [%(levelname)s] %(message)s",

        handlers=[logging.StreamHandler()],

    )

def create_dataset_from_db(cfg: MasterConfig, cfg_mod: object, time_range: dict, output_filename: str):

    """

    Creates a dataset NPZ file by finding spike signals in the database

    for a given time range and saving the corresponding session windows.

    """

    start_utc = time_range["start_utc"]

    end_utc = time_range["end_utc"]

    engine = create_engine(cfg.db.dsn)

    if cfg.paper.symbols == "ALL":

        logging.info("`paper.symbols` is 'ALL'. Fetching all available symbols from the database for the given time range...")

        try:

            with engine.connect() as conn:

                query = text("SELECT DISTINCT symbol FROM klines_1m WHERE open_time_ms >= :start_ts AND open_time_ms < :end_ts AND is_closed = TRUE")

                result = conn.execute(query, {

                    "start_ts": int(pd.to_datetime(start_utc).timestamp() * 1000),

                    "end_ts": int(pd.to_datetime(end_utc).timestamp() * 1000)

                })

                symbols = [row[0] for row in result]

        except Exception as e:

            logging.error(f"Failed to fetch all symbols from DB: {e}", exc_info=True)

            return

    elif isinstance(cfg.paper.symbols, list):

        symbols = cfg.paper.symbols

    else:

        logging.error("`cfg.paper.symbols` must be a list or 'ALL'.")

        return

    if not symbols:

        logging.error("Symbol list is empty. Aborting dataset creation.")

        return

    logging.info(f"Scanning for spike signals from {start_utc} to {end_utc} for {len(symbols)} symbols...")

    try:

        with engine.connect() as conn:

            detector_cfg = cfg.detector

            query = text(f"""

WITH minute_returns AS (

    SELECT 

        open_time_ms,

        symbol, 

        close_price AS close, 

        (close_price / LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY open_time_ms)) - 1 AS ret

    FROM klines_1m 

    WHERE symbol = ANY(:symbols) AND open_time_ms >= :start_ts AND open_time_ms < :end_ts AND is_closed = TRUE

),

rolling_stats AS (

    SELECT 

        open_time_ms, 

        symbol,

        ABS( (close_price / AVG(close_price) OVER (PARTITION BY symbol ORDER BY open_time_ms ROWS BETWEEN {detector_cfg.context_minutes + detector_cfg.window_minutes} PRECEDING AND {detector_cfg.window_minutes} PRECEDING )) - 1 ) AS abs_change,

        AVG(ABS(ret)) OVER (PARTITION BY symbol ORDER BY open_time_ms ROWS BETWEEN {detector_cfg.context_minutes + detector_cfg.window_minutes} PRECEDING AND {detector_cfg.window_minutes} PRECEDING) AS avg_abs_ret_pre

    FROM klines_1m 

    WHERE symbol = ANY(:symbols) AND open_time_ms >= :start_ts AND open_time_ms < :end_ts AND is_closed = TRUE

)

SELECT to_timestamp(open_time_ms / 1000.0) AT TIME ZONE 'UTC' AS ts, symbol 

FROM rolling_stats

WHERE abs_change * 100.0 >= :abs_change_pct AND (abs_change / (avg_abs_ret_pre + 1e-9)) >= :contrast_min

ORDER BY ts, symbol;

""")

            found_spikes_df = pd.read_sql(query, conn, params={

                "symbols": symbols,

                "start_ts": int(pd.to_datetime(start_utc).timestamp() * 1000),

                "end_ts": int(pd.to_datetime(end_utc).timestamp() * 1000),

                "abs_change_pct": detector_cfg.abs_change_pct,

                "contrast_min": detector_cfg.contrast_min,

            })

            found_spikes_df['ts'] = pd.to_datetime(found_spikes_df['ts'], unit='s', utc=True)

    except Exception as e:

        logging.error(f"Failed to scan for spikes in database: {e}", exc_info=True)

        return

    logging.info(f"Found {len(found_spikes_df)} potential spike signals. Applying cooldowns and saving to NPZ...")

    last_signal_time = {}

    experiences = {}

    keys_map = {}

    with engine.connect() as conn:

        for index, row in tqdm(found_spikes_df.iterrows(), total=len(found_spikes_df), desc="Processing signals"):

            symbol, signal_dt = row['symbol'], row['ts']

            if signal_dt <= last_signal_time.get(symbol, dt.datetime.min.replace(tzinfo=dt.timezone.utc)):

                continue

            last_signal_time[symbol] = signal_dt + dt.timedelta(minutes=detector_cfg.cooldown_minutes)

            seq_start = signal_dt - dt.timedelta(minutes=cfg.seq.pre_signal_len)

            seq_end = signal_dt + dt.timedelta(minutes=cfg.seq.post_signal_len)

            start_ms = int(seq_start.timestamp() * 1000)

            end_ms = int(seq_end.timestamp() * 1000)

            query_data = text("""

                SELECT 

                    open_time_ms,

                    open_price AS open,

                    high_price AS high,

                    low_price AS low,

                    close_price AS close,

                    base_volume AS volume,

                    quote_volume,

                    trade_count AS num_trades,

                    taker_base_volume AS taker_base,

                    taker_quote_volume AS taker_quote

                FROM klines_1m 

                WHERE symbol = :symbol 

                    AND open_time_ms >= :start_ms 

                    AND open_time_ms < :end_ms 

                    AND is_closed = TRUE 

                ORDER BY open_time_ms ASC;

            """)

            df_signal = pd.read_sql(query_data, conn, params={

                "symbol": symbol, 

                "start_ms": start_ms, 

                "end_ms": end_ms

            })

            if len(df_signal) < cfg.seq.full_seq_len:

                continue

            # Derive ts

            df_signal['ts'] = pd.to_datetime(df_signal['open_time_ms'], unit='ms', utc=True)

            # Resample to fill gaps (causal ffill)

            df_signal = df_signal.set_index('ts').resample('1T').ffill().reset_index()

            # Compute vwap

            vwap = np.where(df_signal['volume'] > 0, df_signal['quote_volume'] / df_signal['volume'], np.nan)

            df_signal['vwap'] = vwap

            df_signal['vwap'] = df_signal['vwap'].fillna(0)

            # Select and rename to standard channels

            channels_df = pd.DataFrame({

                'open': df_signal['open'],

                'high': df_signal['high'],

                'low': df_signal['low'],

                'close': df_signal['close'],

                'volume': df_signal['volume'],  # base_volume

                'quote_volume': df_signal['quote_volume'],

                'num_trades': df_signal['num_trades'],

                'taker_base': df_signal['taker_base'],

                'taker_quote': df_signal['taker_quote'],

                'vwap': df_signal['vwap']

            })

            # Truncate to exact full_seq_len if resample added extras

            channels_df = channels_df.iloc[:cfg.seq.full_seq_len]

            if len(channels_df) == cfg.seq.full_seq_len:

                seq_arr = channels_df.to_numpy(dtype=np.float32)

                key = f"{symbol}_{signal_dt.strftime('%Y%m%d_%H%M%S')}"

                experiences[key] = seq_arr

                keys_map[key] = (symbol, signal_dt)

    output_path = os.path.join("data", output_filename)

    experiences["_keys_map_"] = keys_map

    np.savez_compressed(output_path, **experiences)

    logging.info(f"Successfully created dataset with {len(keys_map)} sessions: {output_path}")

if __name__ == "__main__":

    setup_logging()

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/alpha.py"

    cfg = load_config(config_path)

    # We need to load the module to get the 'data' dictionary

    spec = importlib.util.spec_from_file_location("experiment_cfg", config_path)

    cfg_mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(cfg_mod)

    # Define time ranges for train, validation, and test sets

    # --- NEW: Adjusted periods to match available DB data (Oct 2024 - Sep 2025) ---

    # Train: 8 months

    create_dataset_from_db(cfg, cfg_mod, {"start_utc": "2024-10-01T00:00:00Z", "end_utc": "2025-06-01T00:00:00Z"}, "train_data_fair_8m.npz")

    # Validation: 2 months

    create_dataset_from_db(cfg, cfg_mod, {"start_utc": "2025-06-01T00:00:00Z", "end_utc": "2025-08-01T00:00:00Z"}, "val_data_fair_2m.npz")

    # Test/Backtest: 2 months

    create_dataset_from_db(cfg, cfg_mod, {"start_utc": "2025-08-01T00:00:00Z", "end_utc": "2025-10-01T00:00:00Z"}, "backtest_data_fair_2m.npz")
