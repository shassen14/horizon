import argparse
import sys
from pathlib import Path

from horizon.config import ConfigError, database_url
from horizon.db.migrate import migrate

# Resolves from this file to the repo root; works because the package is
# installed in editable mode from this repo.
MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


def run_migrate() -> int:
    try:
        url = database_url()
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1

    migrations = migrate(url, MIGRATIONS_DIR)

    if not migrations:
        print("already up to date")
    else:
        print(f"applied migrations: {migrations}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horizon")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="apply pending migrations")
    args = parser.parse_args(argv)

    match args.command:
        case "migrate":
            return run_migrate()

    raise AssertionError(args.command)
