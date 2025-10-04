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

    where = ["ts >= EXTRACT(EPOCH FROM %s::timestamptz)::bigint*1000",
             "ts <  EXTRACT(EPOCH FROM %s::timestamptz)::bigint*1000"]
    params = [args.start_utc, args.end_utc]

    if args.symbols != "ALL":
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        where.append("symbol = ANY(%s)")
        params.append(syms)

    sql = f'''
        WITH base AS (
            SELECT symbol, ts, {", ".join(CHANNELS)}
            FROM v_klines_1m_npz
            WHERE {" AND ".join(where)}
        )
        SELECT ts, symbol, {", ".join(CHANNELS)}
        FROM base
        ORDER BY ts, symbol;
    '''

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
        symbol=df["symbol"].values,
        **{ch: df[ch].astype("float32").values for ch in CHANNELS}
    )
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
