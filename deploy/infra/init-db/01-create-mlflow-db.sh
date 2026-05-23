#!/bin/bash
# Runs once, only on first DB init (when the data dir is empty).
# Creates the MLflow backend role + database inside the same Postgres instance.
# Reads MLFLOW_DB_* from the container env (loaded via env_file in compose.yml).
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  CREATE ROLE ${MLFLOW_DB_USER} WITH LOGIN PASSWORD '${MLFLOW_DB_PASSWORD}';
  CREATE DATABASE ${MLFLOW_DB_NAME} OWNER ${MLFLOW_DB_USER};
EOSQL
