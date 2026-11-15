# Horizon

Quantitative trading research platform in Python and SQL: ingests market data into
TimescaleDB, computes features, and trains and certifies ML models.

## Setup

```bash
uv sync                  # venv + editable install, puts `horizon` on the venv PATH
cp .env.example .env     # then fill in DATABASE_URL and other secrets
```

App config lives in `config/default.toml`.

## Database migrations

```bash
uv run --env-file .env horizon migrate
```

Applies pending SQL files from `migrations/` in order and records each one in
`schema_migrations`. Already-applied files are checksummed, so editing one fails
loudly instead of silently diverging.

## Tests

```bash
uv run pytest
```

Requires Docker. DB tests start a throwaway TimescaleDB container via
testcontainers; no `DATABASE_URL` is needed and no real data is touched.

## Checks

```bash
uv run ruff check src tests
uv run mypy src/horizon tests/db
```
