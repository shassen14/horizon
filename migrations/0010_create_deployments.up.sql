-- A deployment is an active PIN: which node version is serving, in which strategy,
-- in which mode, right now. It is the operational source of truth the scheduler reads
-- to know what to run tonight, and the row a promotion updates. It deliberately holds
-- NO identity of its own -- node_kind, the artifact, behavior_version, labels and
-- horizon all live in node_versions (the immutable catalog). A deployment only points
-- at one catalogued version and wraps it in runtime context. This split is what lets
-- the same node swap its live version (or swap a model for a rule) by re-pinning here,
-- with the full history of what each version WAS preserved in node_versions.
--
-- Both models and rules are pinned here -- a rule alpha is a first-class deployment
-- that produces signals, not a config detail hidden in the manifest. node_version is
-- NULL until the first promotion pins a version; node_key is always set so a
-- deployment knows which node it is for even before anything is pinned. The composite
-- FK (node_key, node_version) -> node_versions is added once that table exists (a
-- later migration); with a NULL node_version the FK is simply not enforced, which is
-- the unpinned state.
--
-- role names the OUTPUT CONTRACT the node produces ('asset_scores', 'regime_state',
-- 'vol_forecast', ...). It is open TEXT on purpose: a new contract is a new value, not
-- a migration. Which role actually drives trades (the alpha) is a RUNTIME rule the
-- decision layer enforces, not a schema rule -- the DB only guarantees that trade
-- context is internally consistent (see the CHECKs below).
CREATE TABLE deployments (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  strategy_id BIGINT,
  node_key TEXT NOT NULL,
  node_version TEXT,
  role TEXT NOT NULL,
  execution_mode TEXT CHECK (execution_mode IN ('advisory', 'paper', 'live', 'shadow')),
  visibility TEXT NOT NULL DEFAULT 'private' CHECK (visibility IN ('public', 'private')),
  broker_account_ref TEXT,
  pinned_at TIMESTAMPTZ,
  is_active BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now (),
  FOREIGN KEY (strategy_id) REFERENCES strategies (id),
  -- Trade context is all-or-nothing: a traded deployment has BOTH a strategy and an
  -- execution mode; an input-producing node has NEITHER. This is a pure data-validity
  -- invariant (true under any pipeline wiring), so it belongs here.
  CONSTRAINT trade_context_paired CHECK ((strategy_id IS NULL) = (execution_mode IS NULL)),
  -- A broker account is meaningful only when orders actually reach a broker.
  -- advisory/shadow generate no orders, so they hold no broker ref.
  CONSTRAINT broker_only_when_executing CHECK (broker_account_ref IS NULL OR execution_mode IN ('paper', 'live'))
);

-- Enforces the "one champion per audience" invariant: at most one active, non-shadow
-- traded deployment per (strategy, visibility). Shadows (challengers) are unlimited;
-- untraded nodes have NULL execution_mode and fall outside the filter (NULL <> 'shadow'
-- is not true). The predicate names no contract, so the schema carries zero
-- pipeline-wiring assumptions.
CREATE UNIQUE INDEX idx_deployments_one_champion
  ON deployments (strategy_id, visibility)
  WHERE is_active AND execution_mode <> 'shadow';
