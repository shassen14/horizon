-- A strategy is a stable IDENTITY for a trade-decision policy -- a durable name
-- that deployments bind to and signals trace back through. It deliberately holds
-- NO policy parameters (max_positions, allocation caps, entry percentiles): those
-- are node config for the decision stages and live in the versioned manifest
-- (pipeline_configs), so they are single-sourced and tracked over time rather
-- than mutated in place here with no history. Keeping only the identity is what
-- lets the same strategy run different parameter sets at different config
-- versions without the row lying about what was live.
CREATE TABLE strategies (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  description TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now ()
);
