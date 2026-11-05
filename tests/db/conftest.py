from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from testcontainers.community.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    # Timescale's background workers connect to databases, and CREATE DATABASE
    # fails if the database it copies from has other connections.
    container = PostgresContainer(
        "timescale/timescaledb:latest-pg16",
        driver=None,
    ).with_command("postgres -c timescaledb.max_background_workers=0")

    with container as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def empty_db(postgres_url: str) -> str:
    name = f"test_{uuid4().hex}"

    # CREATE DATABASE can't run inside a transaction.
    with psycopg.connect(postgres_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))

    return make_conninfo(postgres_url, dbname=name)
