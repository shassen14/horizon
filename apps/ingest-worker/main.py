# apps/ingest-worker/main.py

import asyncio
import argparse
from packages.quant_lib.logging import LogManager
from sources.alpaca import AlpacaSource
from engine import IngestionEngine


async def main():
    parser = argparse.ArgumentParser(description="Horizon Ingestion Worker")
    parser.add_argument(
        "--mode", choices=["all", "metadata", "daily", "intraday"], default="all"
    )
    args = parser.parse_args()

    # 1. Dependency: Logging
    # We initialize the LogManager once.
    log_manager = LogManager(service_name="ingest-worker")
    root_logger = log_manager.get_logger("main")

    root_logger.info(f"Initializing Ingest Worker in mode: {args.mode}")

    try:
        # 2. Dependency: Data Source
        # We pass the logger to the source too, so it can log its own connection details
        source_logger = log_manager.get_logger("alpaca-source")
        source = AlpacaSource()  # You could update AlpacaSource to accept source_logger

        # 3. Dependency Injection: The Engine
        # We inject the Source, Settings, and a specific Logger into the Engine
        engine_logger = log_manager.get_logger("ingestion-engine")

        engine = IngestionEngine(source=source, logger=engine_logger)

        # 4. Execution
        # The engine is stateful. We must run metadata first to populate the asset map.
        await engine.run_metadata_sync()

        if args.mode in ["all", "daily"]:
            await engine.run_daily_ingestion()

        if args.mode in ["all", "intraday"]:
            await engine.run_intraday_ingestion()

    except Exception as e:
        root_logger.exception("CRITICAL FAILURE in Ingest Worker Main Loop")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
