-- Wires the tape and the runtime pin to the node catalog, now that node_versions
-- exists. Split into its own migration because predictions and deployments are created
-- earlier than node_versions, so the FK cannot be declared inline -- the same ordering
-- pattern already used to attach pipeline_runs to pipeline_configs.
--
-- The FK is composite (node_key, node_version) -> node_versions (node_key, version):
-- it both guarantees the referenced version exists AND that the node_key carried on
-- the row matches the version's node, so a row can never attribute an output to the
-- wrong node. On deployments a NULL node_version (an unpinned deployment) leaves the
-- FK unenforced, which is the intended pre-promotion state.
ALTER TABLE predictions
  ADD CONSTRAINT predictions_node_version_fkey
  FOREIGN KEY (node_key, node_version) REFERENCES node_versions (node_key, version);

ALTER TABLE deployments
  ADD CONSTRAINT deployments_node_version_fkey
  FOREIGN KEY (node_key, node_version) REFERENCES node_versions (node_key, version);
