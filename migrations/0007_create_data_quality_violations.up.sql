-- Output of the data-quality phase (runs after ingest, before features). One
-- row per detected violation. severity gates the pipeline: a 'blocking' row
-- stops downstream jobs; 'warning'/'info' are recorded only.
--
-- asset_id is NULLABLE because some checks are universe-level (e.g. breadth or
-- gap detection across the whole universe) and have no single asset. details is
-- a JSONB payload carrying check-specific context (display-only, never joined
-- on). pipeline_run_id links the violation to the run that found it.
CREATE TABLE data_quality_violations (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  time TIMESTAMPTZ NOT NULL,
  asset_id INT,
  check_name TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('blocking', 'warning', 'info')),
  details JSONB NOT NULL,
  pipeline_run_id BIGINT,
  resolved_at TIMESTAMPTZ,
  resolution_note TEXT,
  FOREIGN KEY (asset_id) REFERENCES assets (id),
  FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (id)
);

-- Gate query: before running downstream jobs the pipeline checks for unresolved
-- 'blocking' rows, so filter by severity and open (resolved_at IS NULL) state.
CREATE INDEX idx_dq_violations_severity_resolved ON data_quality_violations (severity, resolved_at);

-- Per-asset history: "all violations for asset X over time".
CREATE INDEX idx_dq_violations_asset_time ON data_quality_violations (asset_id, time);
