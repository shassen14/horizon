ALTER TABLE market_data_daily ADD CONSTRAINT market_data_daily_utc_midnight CHECK (
  time = date_trunc ('day', time AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
);
