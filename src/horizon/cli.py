import argparse
import sys
from pathlib import Path

from horizon.config import ConfigError, database_url
from horizon.db.migrate import migrate

# Resolves from this file to the repo root; works because the package is
# installed in editable mode from this repo.
MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"


# Returns the process exit code: 0 on success, 1 on a problem the user can fix.
# Errors go to stderr so stdout stays clean for anything reading it.
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


# argv is None when run as the `horizon` script, so argparse reads sys.argv[1:];
# tests pass a list instead. On bad input argparse exits with code 2 by itself.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horizon")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="apply pending migrations")
    args = parser.parse_args(argv)

    match args.command:
        case "migrate":
            return run_migrate()

    # argparse rejects unknown commands, so this means a subcommand was registered
    # above without a matching case.
    raise AssertionError(args.command)
