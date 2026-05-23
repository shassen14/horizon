-- Market-wide regime inputs: one row PER DAY, no asset dimension (unlike
-- market_data_daily). This is the regime model's fuel and also supplies context
-- features to the alpha model.
--
-- Two numeric scales live here, by convention: bounded oscillators (RSI/ADX)
-- and breadth percentages are stored 0-100 as NUMERIC(5,2) -- their canonical
-- domain; changes/returns (*_pct_change_*) are stored as decimal fractions
-- (e.g. +5% -> 0.05). The change columns are NULLABLE because the earliest rows
-- in history have no prior day to diff against.
--
-- Hypertable on time. Compression deliberately has NO compress_segmentby: there
-- is no repeating categorical (no asset_id) to group rows by, so plain columnar
-- compression of each time chunk is all that applies.
CREATE TABLE market_context_daily (
  time TIMESTAMPTZ NOT NULL,
  vix_close NUMERIC(8, 4) NOT NULL,
  vix_pct_change_1d NUMERIC(8, 4),
  breadth_pct_above_sma20 NUMERIC(5, 2),
  breadth_pct_above_sma50 NUMERIC(5, 2),
  breadth_pct_above_sma200 NUMERIC(5, 2),
  advance_decline_ratio NUMERIC(8, 4),
  spy_rsi_14 NUMERIC(5, 2),
  spy_adx_14 NUMERIC(5, 2),
  hy_credit_spread NUMERIC(8, 4),
  credit_spread_pct_change_5d NUMERIC(8, 4),
  ten_year_yield NUMERIC(8, 4),
  tlt_pct_change_21d NUMERIC(8, 4),
  PRIMARY KEY (time)
);

SELECT
  create_hypertable ('market_context_daily', by_range ('time'));

ALTER TABLE market_context_daily
SET
  (timescaledb.compress);

SELECT
  add_compression_policy ('market_context_daily', INTERVAL '7 days');
