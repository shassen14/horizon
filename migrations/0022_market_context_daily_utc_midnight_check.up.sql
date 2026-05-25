-- market_context_daily is one row per trading day (PK is `time` alone) and aligns
-- with market_data_daily / features_daily on that key. Same invariant: `time` is
-- the trading DATE stored at 00:00:00 UTC. Without it, a non-normalized timestamp
-- becomes a second, non-colliding row for the same day that no longer joins the
-- rest of that day's data -- the regime model would then read context that does
-- not line up with the bars and features it is supposed to describe.
ALTER TABLE market_context_daily ADD CONSTRAINT market_context_daily_utc_midnight CHECK (
  time = date_trunc ('day', time AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
);
