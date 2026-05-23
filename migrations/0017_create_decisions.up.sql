-- The veto/gate tape: one row per (candidate, gating node) verdict. This is the
-- decision layer's half of per-node observability -- without it a risk filter that
-- drops a trade leaves NO trace, so its vetoes can never be audited or counterfactually
-- evaluated ("did this gate dodge losers or kill winners?"). Every candidate a gate
-- judges is recorded, pass OR fail, so both the numerator (vetoes) and the denominator
-- (everything evaluated) are queryable.
--
-- A per-gate LEDGER (one row per gate per candidate), not one terminal row per
-- candidate: each control node writes its own verdict, uniform with how each predictive
-- node writes its own tape, and it assumes no fixed gate order. A surviving candidate
-- has a passed=true row from every gate it cleared PLUS a signals row; a vetoed
-- candidate has a passed=false row and no signals row. decisions and signals share
-- lineage (asset, time, run) but carry no FK between them -- decisions is the full
-- candidate journey, signals is the survivors that became recommendations.
--
-- node_key + node_version reference node_versions: a gating rule is a registered node
-- version (node_kind='rule') exactly like a rule alpha, so control and predictive nodes
-- ride one identity/observability spine. Its config VALUES still live in the manifest
-- (pipeline_configs); node_versions only records that the version exists. reason is the
-- gate's machine-readable cause when it vetoes ('max_position_pct', 'killswitch',
-- 'cost_exceeds_alpha'); details is a display-only JSONB sidecar, never joined on.
--
-- Plain table, not a hypertable: only candidates (the handful that crossed signal_gen),
-- not the whole universe, reach the gates -- tens of rows per day, the same volume
-- reasoning that keeps signals and pipeline_runs relational.
CREATE TABLE decisions (
  time TIMESTAMPTZ NOT NULL,
  asset_id INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  node_version TEXT NOT NULL,
  passed BOOLEAN NOT NULL,
  reason TEXT,
  pipeline_run_id BIGINT,
  details JSONB,
  FOREIGN KEY (asset_id) REFERENCES assets (id),
  FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (id),
  FOREIGN KEY (node_key, node_version) REFERENCES node_versions (node_key, version),
  PRIMARY KEY (time, asset_id, node_key, node_version),
  -- A veto must name its cause so the reason is never lost; a pass needs none (any
  -- pass-time context goes in details, not reason).
  CONSTRAINT veto_has_reason CHECK (passed OR reason IS NOT NULL)
);

-- Per-gate history: "all of this gate's verdicts over time" (e.g. risk_filter's vetoes).
CREATE INDEX idx_decisions_node_time ON decisions (node_key, time);
