# apps/ingest-worker/main.py

import asyncio
import sys
import argparse
from sources.alpaca import AlpacaSource

# Import the logic from jobs.py
from jobs import run_metadata_sync, run_daily_ingestion, run_intraday_ingestion


async def main():
    parser = argparse.ArgumentParser(description="Horizon Ingestion Worker")
    parser.add_argument(
        "--mode",
        choices=["all", "metadata", "daily", "intraday"],
        default="all",
        help="Which ingestion phase to run.",
    )
    args = parser.parse_args()

    source = AlpacaSource()

    # 1. Metadata Phase (Always returns active assets)
    # Even if mode is 'daily', we need the asset map.
    # The 'run_metadata_sync' has built-in caching, so it's cheap to call.
    active_assets = await run_metadata_sync(source)

    if args.mode in ["all", "metadata"]:
        # We already ran it above to get the map, so we are done with this phase.
        pass

    if args.mode in ["all", "daily"]:
        await run_daily_ingestion(source, active_assets)

    if args.mode in ["all", "intraday"]:
        await run_intraday_ingestion(source, active_assets)


if __name__ == "__main__":
    asyncio.run(main())
