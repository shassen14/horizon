-- Every hypertable was created on Timescale's default 7-day chunk interval. At daily
-- grain that yields hundreds of tiny chunks per year, each a separate physical table
-- with its own indexes, catalog entries, and compression metadata -- planning and
-- per-chunk overhead start to rival the data itself. set_chunk_time_interval only
-- affects chunks created afterward, so this must land before bulk ingest; changing it
-- later means dumping and reloading every table.
--
-- market_data_daily: ~1,000 symbols x 252 trading days = ~252,000 rows per 1-year
-- chunk (~25 MB uncompressed). Still small, but multi-year chunks would weaken time
-- pruning for the common "last N months" query.
SELECT
  set_chunk_time_interval ('market_data_daily', INTERVAL '1 YEAR');

-- features_daily: same (asset, day) grain as market_data_daily, so ~252,000 rows per
-- 1-year chunk; rows are wider (~50 MB uncompressed), which is still well within one chunk.
SELECT
  set_chunk_time_interval ('features_daily', INTERVAL '1 YEAR');

-- predictions: one row per (asset, day, node_version). ~252,000 x ~3 live node
-- versions = ~750,000 rows per 1-year chunk.
SELECT
  set_chunk_time_interval ('predictions', INTERVAL '1 YEAR');

-- market_context_daily: one row per trading day with no asset dimension, so a 1-year
-- chunk would hold only ~252 rows. 5 years x 252 = ~1,260 rows per chunk; anything
-- finer is pure overhead.
SELECT
  set_chunk_time_interval ('market_context_daily', INTERVAL '5 YEAR');
