CREATE SCHEMA IF NOT EXISTS rlref;
CREATE TABLE IF NOT EXISTS rlref.klines_1m (
  symbol      TEXT        NOT NULL,
  open_time   TIMESTAMPTZ NOT NULL,
  close_time  TIMESTAMPTZ NOT NULL,
  open        NUMERIC(18,8)  NOT NULL,
  high        NUMERIC(18,8)  NOT NULL,
  low         NUMERIC(18,8)  NOT NULL,
  close       NUMERIC(18,8)  NOT NULL,
  volume      NUMERIC(28,10) NOT NULL,
  trades      INTEGER        NOT NULL,
  event_time  TIMESTAMPTZ    NOT NULL,
  is_closed   BOOLEAN        NOT NULL,
  PRIMARY KEY (symbol, open_time)
);
CREATE INDEX IF NOT EXISTS ix_klines_1m_time ON rlref.klines_1m(open_time DESC);

CREATE TABLE IF NOT EXISTS rlref.agg_trades (
  symbol      TEXT         NOT NULL,
  agg_id      BIGINT       NOT NULL,
  price       NUMERIC(18,8)  NOT NULL,
  quantity    NUMERIC(28,10) NOT NULL,
  first_id    BIGINT       NOT NULL,
  last_id     BIGINT       NOT NULL,
  trade_time  TIMESTAMPTZ  NOT NULL,
  is_maker    BOOLEAN      NOT NULL,
  event_time  TIMESTAMPTZ  NOT NULL,
  PRIMARY KEY (symbol, agg_id)
);
CREATE INDEX IF NOT EXISTS ix_agg_trades_time ON rlref.agg_trades(trade_time DESC);
-- (опционально) TimescaleDB:
-- CREATE EXTENSION IF NOT EXISTS timescaledb;
-- SELECT create_hypertable('rlref.klines_1m','open_time',if_not_exists=>TRUE);
-- SELECT create_hypertable('rlref.agg_trades','trade_time',if_not_exists=>TRUE);
