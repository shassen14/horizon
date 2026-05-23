-- TimescaleDB must be enabled before any hypertable is created. This runs first
-- so the extension exists for every table that follows.
CREATE EXTENSION IF NOT EXISTS timescaledb;
