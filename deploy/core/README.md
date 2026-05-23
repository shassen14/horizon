# Horizon core stack

The always-on Rust services plus the public tunnel:

- **api** — `horizon-api` binary (Axum), serves predictions/trade data
- **scheduler** — `horizon-scheduler` binary, market-aware job runner
- **tunnel** — `cloudflared`, exposes the API via a Cloudflare named tunnel

Both binaries build from one image (`horizon-core:latest`); `scheduler` reuses the image
`api` builds. Runs on any always-on host — it's stateless and points at the infra stack.

> **Status:** `horizon-api` / `horizon-scheduler` are currently stubs. The image builds and
> the containers run, but they do no real work until those binaries are implemented.

## Prerequisites

- The **infra stack** is up and reachable (see `../infra`).
- A Cloudflare named tunnel + its token.

## First-time setup

```sh
cp .env.example .env      # then edit:
```

- `DATABASE_URL` / `MLFLOW_TRACKING_URI` → point at the infra host (hostname or IP), using
  the same Postgres credentials/port from `infra/.env`.
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` → paper-trading keys.
- `CLOUDFLARE_TUNNEL_TOKEN` → from the tunnel you created.

## Start

```sh
docker compose up -d --build
docker compose logs -f api
docker compose down
```

The API is published on `${API_PORT}` (default 8080). The build context is the repo root,
so the image always reflects the current workspace source.
