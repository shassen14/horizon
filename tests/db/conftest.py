from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from testcontainers.community.postgres import PostgresContainer

from horizon.db.migrate import migrate

MIGRATIONS = Path(__file__).parents[2] / "migrations"


def create_database(postgres_url: str, name: str, template: str | None = None) -> str:
    query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
    if template is not None:
        query += sql.SQL(" TEMPLATE {}").format(sql.Identifier(template))

    # CREATE DATABASE can't run inside a transaction.
    with psycopg.connect(postgres_url, autocommit=True) as conn:
        conn.execute(query)

    return make_conninfo(postgres_url, dbname=name)


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


@pytest.fixture(scope="session")
def template_db(postgres_url: str) -> str:
    name = "horizon_template"
    migrate(create_database(postgres_url, name), MIGRATIONS)
    return name


@pytest.fixture
def migrated_db(postgres_url: str, template_db: str) -> str:
    return create_database(postgres_url, f"test_{uuid4().hex}", template=template_db)


@pytest.fixture
def empty_db(postgres_url: str) -> str:
    return create_database(postgres_url, f"test_{uuid4().hex}")
