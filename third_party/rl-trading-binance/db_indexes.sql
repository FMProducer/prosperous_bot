-- db_indexes.sql: Perf boost for window scans (Index Scan vs Seq; query time /10)
-- Run once; ~5-15 min total (90M rows; pg auto-parallel)

-- Drop if exists (safe recreate)
DROP INDEX IF EXISTS idx_klines_symbol_time_closed;
DROP INDEX IF EXISTS idx_klines_close_price;

-- Main: symbol + time (for PARTITION/ORDER; filtered is_closed for spikes)
CREATE INDEX CONCURRENTLY idx_klines_symbol_time_closed ON klines_1m (symbol, open_time_ms) WHERE is_closed = TRUE;
-- ~2-5 min; CONCURRENTLY no lock.

-- Aux: close_price for AVG (helps window compute)
CREATE INDEX CONCURRENTLY idx_klines_close_price ON klines_1m (symbol, open_time_ms, close_price DESC) WHERE is_closed = TRUE;
-- ~3-10 min; DESC optional (for LAG/AVG).

-- Stats update (optimizer uses indexes)
ANALYZE klines_1m;
-- <1 min.

-- Verify (EXPLAIN: expect "Index Scan using idx_klines_symbol_time_closed"; time <5s)
EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) FROM klines_1m 
WHERE symbol = 'BTCUSDT' AND open_time_ms >= 1727740800000 AND open_time_ms < 1748736000000 AND is_closed = TRUE;
-- If "Seq Scan" (>10s) — data small; else OK.

-- Vacuum (cleanup; optional if recent inserts)
VACUUM ANALYZE klines_1m;
