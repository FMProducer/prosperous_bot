DROP MATERIALIZED VIEW IF EXISTS public.mv_candles_prepared;
CREATE MATERIALIZED VIEW public.mv_candles_prepared AS
SELECT
  symbol,
  (to_timestamp(open_time_ms / 1000.0) AT TIME ZONE 'UTC') AS ts_utc,
  open_price::double precision  AS open,
  high_price::double precision  AS high,
  low_price::double precision   AS low,
  close_price::double precision AS close,
  base_volume::double precision AS volume,
  CASE WHEN base_volume > 0 THEN (quote_volume / base_volume)::double precision ELSE NULL END AS vwap,
  trade_count::integer AS trades
FROM public.klines_1m
WHERE is_closed IS TRUE
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS mv_candles_prepared_uq
ON public.mv_candles_prepared (symbol, ts_utc);
