# Pinned MLflow + Postgres driver, baked at build time.
# Bump MLFLOW_VERSION deliberately rather than letting `pip install mlflow` float.
FROM python:3.11-slim

ARG MLFLOW_VERSION=3.12.0
ARG PSYCOPG2_VERSION=2.9.10

RUN pip install --no-cache-dir \
      "mlflow==${MLFLOW_VERSION}" \
      "psycopg2-binary==${PSYCOPG2_VERSION}"

EXPOSE 5002
# The actual `mlflow server ...` invocation lives in compose.yml so the
# backend-store-uri can interpolate secrets from .env at runtime.
