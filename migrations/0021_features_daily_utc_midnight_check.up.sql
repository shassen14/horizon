-- features_daily is keyed by trading day and joins 1:1 with market_data_daily on
-- `time`. The same invariant that table enforces must hold here: a daily bar's
-- `time` is the trading DATE stored at 00:00:00 UTC. A writer that forgets to
-- normalize a source timestamp would create a row that silently fails to join its
-- source bar (and never collides on the PK to expose the mistake), so the join
-- just drops it. Enforce the invariant at the same boundary the data is written.
ALTER TABLE features_daily ADD CONSTRAINT features_daily_utc_midnight CHECK (
  time = date_trunc ('day', time AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
);
