# apps/feature_worker/main.py

import argparse
import asyncio
from packages.quant_lib.logging import LogManager
from engine import FeatureEngine


async def main():
    # 1. Setup CLI Arguments
    parser = argparse.ArgumentParser(description="Horizon Feature Worker")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full recalculation of all history, ignoring existing data.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Run for a specific symbol only (e.g. 'AAPL')",
    )
    args = parser.parse_args()

    # 2. Initialize Logger
    # We initialize the LogManager once for this application.
    log_manager = LogManager(service_name="feature-worker")
    root_logger = log_manager.get_logger("main")

    root_logger.info("--- Initializing Feature Worker ---")

    try:
        # 3. Dependency Injection: Create the Engine
        # We inject a specific logger context for the engine itself.
        engine_logger = log_manager.get_logger("feature-engine")

        engine = FeatureEngine(logger=engine_logger)

        # 4. Run the engine
        await engine.run(force_full=args.force, symbol=args.symbol)

    except Exception as e:
        # The root logger catches any unhandled exceptions from the engine
        root_logger.exception("CRITICAL FAILURE in Feature Worker Main Loop")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
