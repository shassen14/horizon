# Horizon infra stack

Stateful backing services that the rest of Horizon connects to:

- **database** — TimescaleDB (the app DB + the MLflow backend DB, in one Postgres instance)
- **mlflow** — MLflow tracking server, backed by Postgres, artifacts on disk
- **adminer** — lightweight DB web UI

Host this wherever durable storage lives (NAS, server, VM).

## First-time setup

```sh
cp .env.example .env      # then edit: set real passwords, confirm HORIZON_DATA_ROOT
```

`HORIZON_DATA_ROOT` must point at a path that exists on the host. On first start (empty
data dir) the `init-db` script auto-creates the MLflow role + database — no manual step.

## Start (CLI)

```sh
docker compose up -d --build
docker compose logs -f mlflow      # watch it come up
docker compose down                # stop (data persists in HORIZON_DATA_ROOT)
```

| Service | URL |
|---|---|
| MLflow UI | http://<host>:5002 |
| Adminer   | http://<host>:8080 |
| Postgres  | <host>:5434 |

## Start (Synology Container Manager)

1. Copy this `infra/` folder to the NAS (File Station, git, or a shared folder).
2. Create the `.env` here first (Container Manager reads it from the project folder).
3. Container Manager → **Project** → **Create**.
4. Set **Path** to this folder. Container Manager looks for `docker-compose.yml` — if it
   doesn't detect `compose.yml`, rename the file to `docker-compose.yml` (the CLI accepts
   either name, so this is safe).
5. It will build the MLflow image and start all three services. Manage start/stop from the
   Project view afterward.

## Notes

- Data lives under `HORIZON_DATA_ROOT` (`timescaledb/` and `mlartifacts/`), not in the
  container — it survives `down` and rebuilds.
- MLflow version is pinned in `mlflow/Dockerfile` (`MLFLOW_VERSION`). Bump it deliberately.
- The `init-db` script only runs when the data dir is empty. To re-bootstrap, the data dir
  must be wiped first.
