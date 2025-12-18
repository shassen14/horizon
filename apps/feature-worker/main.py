# apps/feature-worker/main.py

import asyncio
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from engine import FeatureEngine


async def main():
    # 1. Initialize Logger
    # We initialize the LogManager once for this application.
    log_manager = LogManager(service_name="feature-worker")
    root_logger = log_manager.get_logger("main")

    root_logger.info("--- Initializing Feature Worker ---")

    try:
        # 2. Dependency Injection: Create the Engine
        # We inject a specific logger context for the engine itself.
        engine_logger = log_manager.get_logger("feature-engine")

        engine = FeatureEngine(logger=engine_logger)

        # 3. Run the engine
        await engine.run()

    except Exception as e:
        # The root logger catches any unhandled exceptions from the engine
        root_logger.exception("CRITICAL FAILURE in Feature Worker Main Loop")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
