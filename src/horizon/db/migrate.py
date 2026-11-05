import hashlib
from pathlib import Path

import psycopg

MIGRATION_LOCK_ID = 727274


def migrate(conninfo: str, migration_dir: Path) -> list[int]:
    # Autocommit makes conn.transaction() below a real BEGIN/COMMIT, not a savepoint.
    with psycopg.connect(conninfo, autocommit=True) as conn:
        # Taken before reading the tracking table, so a second runner can't act on a
        # stale list. Session-level, so it survives each migration's commit.
        conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
                version INT,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (version)
                )
                     """)

        paths = sorted(migration_dir.glob("*.up.sql"))

        if not paths:
            raise RuntimeError(f"no migrations found in {migration_dir}")

        rows = conn.execute(
            "SELECT version, checksum FROM schema_migrations"
        ).fetchall()

        applied: dict[int, str] = {version: checksum for version, checksum in rows}

        newly_applied: list[int] = []

        for path in paths:
            version_str, rest = path.name.split("_", 1)
            name = rest.removesuffix(".up.sql")
            version = int(version_str)

            sql = path.read_bytes()
            checksum = hashlib.sha256(sql).hexdigest()

            if version in applied:
                if applied[version] != checksum:
                    raise RuntimeError(
                        f"{path.name} changed after it was applied; "
                        "restore it and put the change in a new migration"
                    )
                continue

            # The migration and its tracking row commit together or not at all.
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    """INSERT into schema_migrations
                            (version, name, checksum)
                             VALUES (%s, %s, %s)
                             """,
                    (version, name, checksum),
                )

            newly_applied.append(version)

    return newly_applied
