# scripts/provision_db.py

import asyncio
import sys
import argparse
from pathlib import Path
from sqlalchemy import text

# Hack to make local packages importable without installing them globally
sys.path.append(str(Path(__file__).resolve().parents[1]))

from packages.database.session import get_autocommit_connection
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager

# Simple logger for this script
log_manager = LogManager("provisioner", debug=True)
logger = log_manager.get_logger("main")


async def provision_resource(conn, user, password, db_name, recreate: bool):
    """
    Handles the lifecycle of a single Database/User pair.
    If recreate=True, it drops them first.
    Then, it ensures they exist.
    """

    # 1. Destructive Phase (Only if requested)
    if recreate:
        logger.warning(f"💥 Dropping Database '{db_name}' and User '{user}'...")
        try:
            await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE);"))
            logger.info(f"Database '{db_name}' dropped.")

            await conn.execute(text(f"DROP USER IF EXISTS {user};"))
            logger.info(f"User '{user}' dropped.")
        except Exception as e:
            logger.error(f"Error dropping resources: {e}")
            # Proceeding anyway to try and repair/create

    # 2. Creation Phase (Idempotent)

    # Create User
    user_exists = await conn.scalar(
        text(f"SELECT 1 FROM pg_roles WHERE rolname='{user}'")
    )
    if not user_exists:
        logger.info(f"Creating User: {user}")
        await conn.execute(text(f"CREATE USER {user} WITH PASSWORD '{password}';"))
    else:
        logger.info(f"User '{user}' exists.")

    # Create Database
    db_exists = await conn.scalar(
        text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
    )
    if not db_exists:
        logger.info(f"Creating Database: {db_name}")
        await conn.execute(text(f"CREATE DATABASE {db_name} OWNER {user};"))
    else:
        logger.info(f"Database '{db_name}' exists.")


async def main():
    parser = argparse.ArgumentParser(description="Horizon Database Provisioner")

    # Exclusive group ensures you don't confuse flags (optional)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--recreate-all",
        action="store_true",
        help="⚠️ DESTROYS and recreates ALL databases.",
    )

    parser.add_argument(
        "--recreate-app", action="store_true", help="⚠️ DESTROYS Horizon App DB only."
    )
    parser.add_argument(
        "--recreate-mlflow", action="store_true", help="⚠️ DESTROYS MLflow DB only."
    )

    args = parser.parse_args()

    # Logic to determine what to nuke
    nuke_app = args.recreate_all or args.recreate_app
    nuke_mlflow = args.recreate_all or args.recreate_mlflow

    logger.info(f"Connecting to Postgres at {settings.db.host}...")

    # Connect to default 'postgres' db to perform admin tasks
    async with get_autocommit_connection(db_override="postgres") as conn:

        # 1. Provision Horizon App DB
        logger.info("--- Processing Horizon App DB ---")
        await provision_resource(
            conn,
            settings.db.user,
            settings.db.password,
            settings.db.name,
            recreate=nuke_app,
        )

        # 2. Provision MLflow DB
        logger.info("--- Processing MLflow DB ---")
        await provision_resource(
            conn,
            settings.mlflow.db_user,
            settings.mlflow.db_password,
            settings.mlflow.db_name,
            recreate=nuke_app,
        )
    logger.success("✅ Provisioning Complete.")

    if nuke_app:
        logger.info(
            "⚠️  App DB recreated: Run 'alembic upgrade head' to rebuild tables!"
        )

    if nuke_mlflow:
        logger.info("⚠️  MLflow DB recreated: Restart the MLflow container on NAS.")


if __name__ == "__main__":
    asyncio.run(main())
