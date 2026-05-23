-- Content-addressed snapshot of the pipeline configuration that produced a run:
-- the wiring (which stages, in what order) plus every stage's rule config and
-- its declared behavior_version. This is the non-model half of trade lineage --
-- models are pinned by MLflow run id in deployments; everything else (risk and
-- cost rules, signal thresholds, the architecture itself) is pinned here.
--
-- config_hash is a hash of the CANONICALIZED manifest (comments stripped, keys
-- ordered, whitespace normalized) computed by the runner. Identity is the
-- content, not a git commit -- so a config that never enters git is still
-- tracked, and a comment-or-formatting-only edit does not produce a new hash.
-- Same content = same hash = stored once.
--
-- content holds the canonical manifest as JSONB so a past config can be
-- introspected ("which configs used a 10% drawdown gate?") and replayed exactly.
-- visibility mirrors deployments.visibility: a private architecture is tracked
-- here without ever being shared; a public one can be exported by its hash. This
-- decouples "tracked" from "shared" -- everything is recorded locally, sharing
-- is a separate, deliberate act.
CREATE TABLE pipeline_configs (
  config_hash TEXT PRIMARY KEY,
  content JSONB NOT NULL,
  visibility TEXT NOT NULL DEFAULT 'private' CHECK (visibility IN ('public', 'private')),
  label TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now ()
);

-- Links each run to the exact configuration that produced it. NULLABLE: a
-- bootstrap job that runs under no manifest carries NULL. The runner upserts the
-- config row (ON CONFLICT DO NOTHING) then stamps the run with its hash, so a
-- signal traces back through its run to the precise rules + wiring that fired it.
ALTER TABLE pipeline_runs
  ADD COLUMN config_hash TEXT,
  ADD FOREIGN KEY (config_hash) REFERENCES pipeline_configs (config_hash);
