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

# Construct absolute path to the config file based on the script's location
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
default_config_path = os.path.join(project_root, "config", "export_npz.json")

config_path = os.environ.get("EXPORT_CONFIG", default_config_path)
cfg = json.load(open(config_path, "r"))

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
output_path = os.path.join(project_root, cfg["export"]["out_path"])
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.savez_compressed(output_path, **arr)
print("Saved:", output_path, "rows:", len(ts))