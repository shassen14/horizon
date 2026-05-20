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
