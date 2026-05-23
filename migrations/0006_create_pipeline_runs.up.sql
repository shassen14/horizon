-- Job-execution history for the scheduler: one row per job run. This is
-- observability the scheduler can QUERY ("did today's ingest succeed before I
-- run features?"), which structured logging/tracing cannot answer. Relational,
-- not a hypertable -- a few rows per day, status-driven, never time-range
-- scanned.
--
-- status lifecycle: 'running' on start, then one of 'success' / 'failed'.
-- 'skipped' records a job that correctly did nothing (non-trading day -- a
-- weekend or market holiday) so the scheduler can tell "didn't run" apart from
-- "ran and failed". Add states via a future migration if needed.
CREATE TABLE pipeline_runs (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  job_name TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now (),
  finished_at TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed', 'skipped')),
  error TEXT,
  rows_processed BIGINT NOT NULL DEFAULT 0,
  -- Lineage: the run whose output fed this run (e.g. a features run points at
  -- the ingest run that produced its input bars). NULL for root jobs (ingest).
  -- Self-referential FK so a run can only point at a run that already exists.
  parent_run_id BIGINT,
  FOREIGN KEY (parent_run_id) REFERENCES pipeline_runs (id)
);

-- Hot path: "the last N runs of job X", which the scheduler reads to check a
-- dependency completed before starting the next job.
CREATE INDEX idx_pipeline_runs_job_started ON pipeline_runs (job_name, started_at DESC);
