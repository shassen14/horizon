-- Per-asset technical indicators computed from market_data_daily. Primary ML
-- training input alongside market_context_daily. Wide table (one column per
-- indicator) so one row = one training sample, with no pivoting at query time.
--
-- All indicator columns are NULLABLE on purpose: during an indicator's warmup
-- period (e.g. 200 days of history for SMA-200) there is no valid value yet,
-- and writing a placeholder number would corrupt training.
--
-- features_version pins which feature-computation code produced the row, so a
-- model can refuse to run on rows built by an incompatible feature set
-- (train/serve parity). Hypertable on time; compress_segmentby='asset_id'
-- groups each asset's rows together within a time chunk.
CREATE TABLE features_daily (
  time TIMESTAMPTZ NOT NULL,
  asset_id INTEGER NOT NULL,
  sma_20 NUMERIC(12, 4),
  sma_50 NUMERIC(12, 4),
  ema_12 NUMERIC(12, 4),
  ema_26 NUMERIC(12, 4),
  rsi_14 NUMERIC(5, 2),
  macd NUMERIC(10, 6),
  macd_signal NUMERIC(10, 6),
  macd_hist NUMERIC(10, 6),
  bb_upper NUMERIC(12, 4),
  bb_lower NUMERIC(12, 4),
  bb_pct NUMERIC(5, 2),
  atr_14 NUMERIC(12, 4),
  adx_14 NUMERIC(5, 2),
  roc_5 NUMERIC(10, 6),
  roc_21 NUMERIC(10, 6),
  volume_ratio NUMERIC(8, 4),
  features_version TEXT NOT NULL,
  FOREIGN KEY (asset_id) REFERENCES assets (id),
  PRIMARY KEY (time, asset_id)
);

SELECT
  create_hypertable ('features_daily', by_range ('time'));

ALTER TABLE features_daily
SET
  (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id'
  );

SELECT
  add_compression_policy ('features_daily', INTERVAL '7 days');
