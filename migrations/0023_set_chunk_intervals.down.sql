-- Restore Timescale's default 7-day chunk interval. Like the up migration, this only
-- affects chunks created afterward; existing chunks keep their boundaries.
SELECT
  set_chunk_time_interval ('market_data_daily', INTERVAL '7 days');

SELECT
  set_chunk_time_interval ('features_daily', INTERVAL '7 days');

SELECT
  set_chunk_time_interval ('predictions', INTERVAL '7 days');

SELECT
  set_chunk_time_interval ('market_context_daily', INTERVAL '7 days');
