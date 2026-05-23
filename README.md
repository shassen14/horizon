# Horizon

Self-hosted quantitative trading research platform. Rust core ingests market data,
computes features, and serves an API; a Python pipeline trains and certifies ML models.

## Build

```bash
cargo build --workspace          # Rust crates

cd training && uv sync           # Python pipeline (venv + editable install)
```

## Configuration

- Secrets → `.env` (gitignored; copy from `.env.example`)
- App config → `config/default.toml`
- Model blueprints → `training/config/models/*.yaml`

## Database migrations

Requires the sqlx CLI (one-time install):

```bash
cargo install sqlx-cli --no-default-features --features postgres,rustls
```

With `DATABASE_URL` set in `.env`:

```bash
sqlx migrate run     # apply pending migrations
sqlx migrate info    # show applied/pending status
```
