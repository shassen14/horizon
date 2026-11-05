from pathlib import Path

import psycopg
import pytest

from horizon.db.migrate import migrate

MIGRATIONS = Path(__file__).parents[2] / "migrations"


def test_migrate_applies_then_skips(empty_db: str) -> None:
    first = migrate(empty_db, MIGRATIONS)
    second = migrate(empty_db, MIGRATIONS)
    assert first
    assert second == []


def test_migrate_rejects_edited_file(empty_db: str, tmp_path: Path) -> None:
    migration = tmp_path / "0001_test.up.sql"
    migration.write_text("CREATE TABLE t (id INT);")

    migrate(empty_db, tmp_path)

    migration.write_text("CREATE TABLE s (name TEXT);")

    with pytest.raises(RuntimeError, match="changed after it was applied"):
        migrate(empty_db, tmp_path)


def test_failed_migration_rolls_back(empty_db: str, tmp_path: Path) -> None:
    # Postgres runs a multi-statement string as one implicit transaction, so this
    # passes even without conn.transaction(). The next test covers that.
    migration = tmp_path / "0001_bad.up.sql"
    migration.write_text("CREATE TABLE t (id INT); SELECT 1/0;")

    with pytest.raises(psycopg.errors.DivisionByZero):
        migrate(empty_db, tmp_path)

    with psycopg.connect(empty_db) as conn:
        table = conn.execute("SELECT to_regclass('t')").fetchone()
        assert table == (None,)

        count = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()
        assert count == (0,)


def test_failed_tracking_insert_rolls_back_migration(
    empty_db: str, tmp_path: Path
) -> None:
    # The file itself succeeds, but dropping the tracking table makes the runner's
    # INSERT fail, so the migration must be undone along with it.
    migration = tmp_path / "0001_drops_tracking.up.sql"
    migration.write_text("CREATE TABLE t (id INT); DROP TABLE schema_migrations;")

    with pytest.raises(psycopg.errors.UndefinedTable):
        migrate(empty_db, tmp_path)

    with psycopg.connect(empty_db) as conn:
        table = conn.execute("SELECT to_regclass('t')").fetchone()
        assert table == (None,)
