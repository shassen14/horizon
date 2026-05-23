-- Raw OHLCV daily bars: one row per asset per trading day, never updated after
-- write (the first fetch is canonical). source records provenance -- 'alpaca',
-- 'yfinance', or 'manual'; adjustment records the price-adjustment mode at fetch
-- time -- 'split_div', 'split', or 'none'.
--
-- Foreign Key makes a certain column reference another
-- column from another table
-- Numeric (precision, scale)
-- precision - total significant digits
-- scale - total digits after the decimal
-- Numeric doesn't have floating point approximation, it's exact
-- BIGINT is 64-bit and volume > 2^32 ~ 2.1 billion
CREATE TABLE market_data_daily (
  time TIMESTAMPTZ NOT NULL,
  asset_id INTEGER NOT NULL,
  open NUMERIC(12, 4) NOT NULL,
  high NUMERIC(12, 4) NOT NULL,
  low NUMERIC(12, 4) NOT NULL,
  close NUMERIC(12, 4) NOT NULL,
  volume BIGINT NOT NULL,
  -- Volume-weighted average price. NULLABLE on purpose: a true daily VWAP needs
  -- intraday trade data, so only sources that supply it (e.g. Alpaca) populate
  -- this. Daily-only fallbacks (e.g. YFinance) leave it NULL rather than store a
  -- typical-price approximation that would masquerade as a real VWAP.
  vwap NUMERIC(12, 4),
  source TEXT NOT NULL,
  adjustment TEXT NOT NULL,
  FOREIGN KEY (asset_id) REFERENCES assets (id),
  PRIMARY KEY (time, asset_id)
);

-- converts regular table to time-partitioned hypertable,
-- chunked by time column
SELECT
  create_hypertable ('market_data_daily', by_range ('time'));

-- compress_segmentby - within each time chunk, group
-- rows by 'asset_id' (i.e. GME stays with GME)
ALTER TABLE market_data_daily
SET
  (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id'
  );

SELECT
  add_compression_policy ('market_data_daily', INTERVAL '7 days');
