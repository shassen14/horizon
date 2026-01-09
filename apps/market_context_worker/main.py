import asyncio
import sys
import argparse
from apps.market_context_worker.engine import MarketContextEngine
from packages.quant_lib.logging import LogManager


async def main():
    # 1. Setup Logging
    log_manager = LogManager("market-context-worker", debug=True)
    logger = log_manager.get_logger("main")

    # 2. Parse Arguments
    # We use argparse for robust flag handling
    parser = argparse.ArgumentParser(description="Horizon Market Context Worker")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full recalculation of history (slow, chunked processing).",
    )
    parser.add_argument("--chunk-size", type=int, default=90, help="Days per chunk.")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Parallel DB queries."
    )
    args = parser.parse_args()

    # 3. Initialize Engine
    engine = MarketContextEngine(logger)

    try:
        # 4. Run Logic
        if args.force:
            logger.warning("Force Full Mode enabled. This will take time.")

        await engine.run(
            force_full=args.force,
            chunk_size_days=args.chunk_size,
            concurrency=args.concurrency,
        )

    except Exception as e:
        logger.error(f"Market Context Worker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
