-- Content-addressed recipe for how a node's training TARGETS were produced -- the
-- "ground truth" half of model lineage. Regime has no observable ground truth, so
-- labels are MANUFACTURED by a labeler (HMM, GMM, k-means, a Bayesian mixture);
-- alpha's labels are a deterministic formula (sign of forward return). Both are just
-- "a spec that produced targets", so both live here, addressed by content.
--
-- label_hash is a hash of the CANONICALIZED spec in content (keys ordered,
-- whitespace normalized), computed by the trainer -- identity is the recipe, not a
-- git commit, so a labeler that never enters git is still tracked and an identical
-- recipe is stored once. This is the join key that answers the question that made
-- past regime work irreproducible: did two model versions train on the SAME targets
-- (same label_hash) or did the targets themselves move?
--
-- content holds the canonical recipe as JSONB: the method and its hyperparameters
-- (e.g. {n_states:3, covariance:'full', seed:42}), the fit window, and which inputs
-- it consumed. Enough to REPRODUCE the labels; the fitted labeler artifact and the
-- label series themselves live in MLflow (see mlflow_ref), not here.
-- label_kind is open TEXT (e.g. 'forward_return', 'hmm', 'gmm', 'kmeans') so a new
-- labeling method is a new value, not a migration.
CREATE TABLE label_definitions (
  label_hash TEXT PRIMARY KEY,
  content JSONB NOT NULL,
  label_kind TEXT NOT NULL,
  -- MLflow run holding the fitted labeler + the produced label series. NULL when the
  -- recipe is a pure formula (e.g. forward return) with no fitted artifact to store.
  mlflow_ref TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now ()
);
