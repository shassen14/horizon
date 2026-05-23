-- The categorical-STATE tape: a node's read on what market regime we are in. Market-
-- wide, so there is NO asset_id -- one distribution per (day, node version), unlike
-- predictions which is per-asset. This is the second output kind (alongside the
-- cross-sectional score), and a regime node is interchangeable the same way: a rule
-- (VIX/breadth thresholds) and a model (an ONNX classifier predicting an HMM's labels)
-- both write the identical shape here.
--
-- Store the full DISTRIBUTION, derive the label: one row per (day, node, regime), each
-- carrying that class's probability. The hard "today's regime" is the argmax -- a
-- query, not a stored column -- because the soft probabilities are the richer form and
-- are what later soft-weighting needs; storing only the winning label would throw away
-- information that cannot be recovered. The probabilities for a (day, node) should sum
-- to ~1, but that is a cross-row property SQL cannot CHECK, so it is a writer/data-
-- quality responsibility, not a column constraint.
--
-- Plain table, not a hypertable: regime is a handful of rows per day (a few classes
-- times a few nodes), so it is status-driven and small, never time-range scanned at
-- volume -- the same reasoning that keeps pipeline_runs relational.
CREATE TABLE regime_states (
  time TIMESTAMPTZ NOT NULL,
  node_key TEXT NOT NULL,
  node_version TEXT NOT NULL,
  regime_label TEXT NOT NULL,
  probability NUMERIC(5, 4) NOT NULL CHECK (probability BETWEEN 0 AND 1),
  pipeline_run_id BIGINT,
  FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (id),
  FOREIGN KEY (node_key, node_version) REFERENCES node_versions (node_key, version),
  PRIMARY KEY (time, node_key, node_version, regime_label)
);

-- Per-node history: "this regime node's read over time".
CREATE INDEX idx_regime_states_node_time ON regime_states (node_key, time);
