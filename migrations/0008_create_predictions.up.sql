-- The cross-sectional SCORE tape: one row per (day, asset, node version). This is the
-- canonical alpha contract -- every node that ranks the universe, whether a model or a
-- rule, normalizes its native output (a price, a return, a probability) to the same
-- bounded score here. That normalization is what makes rule and model implementations
-- interchangeable: the decision layer reads a comparable score and never knows or
-- cares which produced it.
--
-- score ONLY -- deliberately not direction or confidence. DIRECTION is a decision, not
-- a prediction: signal_gen derives long/short/flat by thresholding the score, and it
-- lives on signals.direction. CONFIDENCE is meaningful only with uncertainty
-- quantification (ensemble/conformal/quantile); a plain GBM emits none, and anything
-- derived from score (e.g. |score-0.5|) would be redundant -- add a real column if/when
-- UQ arrives. What the score MEANS (P(outperform) vs normalized return) is recorded in
-- its lineage (node_versions -> label_definitions), not duplicated on the tape. The
-- prediction HORIZON is likewise a property of the node version (node_versions.
-- horizon_days), not the per-row output, so it is not repeated here.
--
-- Keyed on the NODE, not the model: node_key + node_version are part of the PRIMARY
-- KEY so multiple nodes AND multiple versions can score the same asset on the same day
-- side by side -- exactly what lets a champion (live) and a shadow challenger both
-- write here without collision. node_version is the surrogate from node_versions; for
-- a model it resolves to an MLflow run, for a rule to its behavior_version. The FK to
-- node_versions is added once that table exists (a later migration), matching the
-- existing pattern for cross-table lineage. node_key is carried (not just derivable
-- via the version) so it can drive compression segmentby and node-scoped queries
-- without a join; the composite FK keeps it honest.
--
-- score is the raw model probability/rank, stored 0-1 (NOT scaled to 0-100) and kept
-- exact as NUMERIC to preserve ranking precision across the universe where third/fourth
-- decimals decide ordering. key_drivers is a display-only JSONB sidecar (SHAP/top
-- features, may be skipped for speed), never joined on. Hypertable on time;
-- compress_segmentby='asset_id, node_key' because queries filter by both.
CREATE TABLE predictions (
  time TIMESTAMPTZ NOT NULL,
  asset_id INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  node_version TEXT NOT NULL,
  score NUMERIC(5, 4) NOT NULL CHECK (score BETWEEN 0 AND 1),
  key_drivers JSONB,
  pipeline_run_id BIGINT,
  FOREIGN KEY (asset_id) REFERENCES assets (id),
  FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (id),
  PRIMARY KEY (time, asset_id, node_key, node_version)
);

SELECT
  create_hypertable ('predictions', by_range ('time'));

ALTER TABLE predictions
SET
  (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id, node_key'
  );

SELECT
  add_compression_policy ('predictions', INTERVAL '7 days');
