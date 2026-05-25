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

## Tests

```bash
cargo test --workspace
```

Integration tests use `#[sqlx::test]`: each spins up a fresh, ephemeral database,
runs all migrations against it, hands the test a pool, then drops it. They need
`DATABASE_URL` pointing at a server allowed to create databases — no real data is
touched.

## Offline builds (`SQLX_OFFLINE`)

sqlx's `query!` macros normally verify SQL against a live database at compile
time. The committed `.sqlx/` directory caches those checks so builds (and CI) work
without a database:

```bash
SQLX_OFFLINE=true cargo build --workspace   # build from cached query metadata
cargo sqlx prepare --workspace              # regenerate .sqlx after editing any query!
```

Regenerate and commit `.sqlx/` whenever you add or change a `query!`/`query_scalar!`.
