-- A signal is the system's actionable recommendation ("BUY AAPL") plus the
-- human's decision on it. It is produced by a specific DEPLOYMENT (a node
-- version running in a mode), not a strategy directly -- so we can trace which
-- node version and audience generated it, and filter public vs private via
-- deployments.visibility. The producing node may be a model or a rule; because
-- every alpha normalizes to the same score, signals need not know which.
--
-- Lifecycle (advisory-first; no broker required):
--   pending  -> approved   (you accept; terminal in advisory mode -- nothing is placed)
--   pending  -> rejected   (you decline)
--   pending  -> expired    (not acted on; signal went stale)
--   approved -> filled     (paper/live only: a position was actually recorded)
--   filled   -> closed     (position later exited)
--   any      -> failed     (execution attempt errored; see failure_reason)
CREATE TABLE signals (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  -- Which deployment produced this signal (node version + mode + audience).
  deployment_id BIGINT NOT NULL,
  -- INTEGER to match assets.id (which is INTEGER, not BIGINT).
  asset_id INTEGER NOT NULL,
  -- A signal is actionable: long (1) or short (-1). Neutral never fires a signal.
  direction SMALLINT NOT NULL CHECK (direction IN (-1, 1)),
  -- Asset price at signal generation; the basis for slippage once a fill exists.
  signal_price NUMERIC(12, 4) NOT NULL,
  -- The normalized alpha score (0-1) that crossed the entry threshold. Same units
  -- whether a model or a rule produced it -- the prediction is always normalized.
  -- DELIBERATELY snapshotted here, not normalized away: although it is reconstructable
  -- from the firing prediction via lineage, a signal is an immutable audit record and
  -- predictions is a compressed hypertable that may be down-sampled over time -- so the
  -- exact score that fired the signal is frozen onto the signal itself. Do not remove.
  triggering_score NUMERIC(5, 4) NOT NULL CHECK (triggering_score BETWEEN 0 AND 1),
  -- Lineage to the exact prediction that fired this signal. The prediction's PK
  -- is (time, asset_id, node_key, node_version); asset_id is here, and
  -- node_key + node_version come from deployment_id -> deployments. This
  -- column supplies the remaining piece (the prediction's trading day), so the
  -- full prediction row is reconstructable via a join. Not a single-column FK
  -- because the target key is composite and spans that join.
  prediction_time TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL CHECK (
    status IN (
      'pending',
      'approved',
      'rejected',
      'expired',
      'filled',
      'closed',
      'failed'
    )
  ),
  -- Set only when status = 'failed'. e.g. 'stale_prediction', 'illiquid', 'broker_5xx'.
  failure_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now (),
  -- Terminal-transition timestamps for advisory mode (pending -> one of these).
  -- Exactly one is set once the signal leaves 'pending'. filled_at / closed_at /
  -- failed_at arrive with the deferred execution layer, not needed advisory-only.
  approved_at TIMESTAMPTZ,
  rejected_at TIMESTAMPTZ,
  expired_at TIMESTAMPTZ,
  notes TEXT,
  FOREIGN KEY (deployment_id) REFERENCES deployments (id),
  FOREIGN KEY (asset_id) REFERENCES assets (id)
);

-- Common query: "all pending signals for deployment X" (the approval queue).
CREATE INDEX idx_signals_deployment_status ON signals (deployment_id, status);
