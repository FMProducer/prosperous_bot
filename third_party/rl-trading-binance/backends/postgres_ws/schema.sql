CREATE TABLE IF NOT EXISTS klines_1m (
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
-- Индексы под выборки по времени:
CREATE INDEX IF NOT EXISTS klines_1m_time_idx ON klines_1m (open_time_ms);

CREATE TABLE IF NOT EXISTS agg_trades (
  symbol     text        NOT NULL,
  agg_id     bigint      NOT NULL,         -- a
  price      numeric(38,18) NOT NULL,      -- p
  quantity   numeric(38,18) NOT NULL,      -- q
  first_id   bigint      NOT NULL,         -- f
  last_id    bigint      NOT NULL,         -- l
  trade_time_ms bigint   NOT NULL,         -- T
  is_maker   boolean     NOT NULL,         -- m
  ingest_ts  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (symbol, agg_id)
);
CREATE INDEX IF NOT EXISTS agg_trades_time_idx ON agg_trades (trade_time_ms);