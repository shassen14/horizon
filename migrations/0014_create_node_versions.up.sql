-- Immutable catalog of every defined node version -- model OR rule -- whether or not
-- it was ever deployed. This is the TRAINING/DEFINITION record (what a version is and
-- what it was trained to predict); deployments is the separate RUNTIME record (which
-- version is pinned live right now). Many versions are catalogued here; few are
-- deployed. Every tape row (predictions, regime_states) references a version here, so
-- a stored output is always interpretable: its targets, horizon, and lineage are one
-- join away, no trip to MLflow required.
--
-- version is a surrogate identity token minted by the trainer for one exact
-- (artifact + behavior) combination. It is what the tapes carry. Using a surrogate --
-- not the MLflow run id directly -- is deliberate: two versions can share a model
-- artifact but differ only in post-processing (a changed softmax or entry threshold),
-- and they must be distinguishable because they emit different outputs. The surrogate
-- gives each its own row; artifact_ref + behavior_version record what actually differs.
--
-- node_kind splits the two implementations of the SAME output contract:
--   'model' -- fit to data, served as an ONNX artifact (artifact_ref = its MLflow run);
--             trained against manufactured/observed targets (label_hash set).
--   'rule'  -- hand-specified logic, no artifact and no labels at all. A rule is the
--             label-free fallback when a model can't be certified: same contract,
--             swapped in with zero schema change.
-- behavior_version is present for BOTH: it versions rule logic for rules, and the
-- post-processing (softmax, normalization, thresholds) for models -- the part the
-- model artifact alone does not capture.
CREATE TABLE node_versions (
  version TEXT PRIMARY KEY,
  node_key TEXT NOT NULL,
  node_kind TEXT NOT NULL CHECK (node_kind IN ('model', 'rule')),
  -- MLflow run id of the served artifact (models only).
  artifact_ref TEXT,
  behavior_version TEXT NOT NULL,
  -- The targets this version was trained against. NULL for rules (no training).
  label_hash TEXT,
  -- Prediction horizon in trading days; part of the target's identity (a 5-day and a
  -- 20-day version of the same node are genuinely different things). NULL for rules.
  horizon_days INT,
  training_window TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now (),
  FOREIGN KEY (label_hash) REFERENCES label_definitions (label_hash),
  -- A rule is hand-specified: it has no fitted artifact and no labels. Enforced here
  -- so the catalog can never claim a rule was trained.
  CONSTRAINT rule_has_no_artifact CHECK (node_kind = 'model' OR artifact_ref IS NULL),
  CONSTRAINT rule_has_no_labels CHECK (node_kind = 'model' OR label_hash IS NULL),
  -- A model is a served artifact, so it must name one.
  CONSTRAINT model_has_artifact CHECK (node_kind = 'rule' OR artifact_ref IS NOT NULL),
  -- Lets the tapes carry node_key alongside version and enforce that the two agree via
  -- a composite FK, so a tape row can never reference a version under the wrong node.
  UNIQUE (node_key, version)
);
