-- Symbol master list: the universe of tradeable assets, referenced by foreign
-- keys throughout (market_data_daily, features_daily, predictions, signals...).
--
-- Design notes:
--   id is an auto-incrementing surrogate key; symbol is the natural key, but an
--     integer id is cheaper to reference from every other table's FK.
--   TEXT over VARCHAR -- Postgres has no length penalty, so don't cap arbitrarily.
--   UNIQUE on symbol so the same ticker can't be registered twice.
--   TIMESTAMPTZ over TIMESTAMP -- stores UTC with offset, avoids timezone ambiguity.
CREATE TABLE assets (
  -- Primary key
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  -- Each asset has an identifier such as GME
  symbol TEXT NOT NULL UNIQUE,
  -- Asset name
  name TEXT,
  -- Platform where one buys or sells the asset
  exchange TEXT,
  -- Investment type such as stock, ETF, mutual fund, etc
  asset_type TEXT NOT NULL DEFAULT 'stock',
  -- Soft delete for assets not being used right now
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  -- Creation Time
  created_at TIMESTAMPTZ NOT NULL DEFAULT now ()
);
