# apps/ingest_worker/main.py

import asyncio
import argparse
from datetime import datetime
from aiolimiter import AsyncLimiter
from packages.quant_lib.helpers.dates import ensure_utc_timestamp
from packages.quant_lib.market_clock import MarketClock

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from engine import IngestionEngine
from sources.factory import get_data_source


async def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Horizon Ingestion Worker")
    parser.add_argument(
        "--mode",
        choices=["all", "metadata", "daily", "intraday"],
        default="all",
        help="Ingestion phase to run.",
    )
    parser.add_argument(
        "--force-metadata",
        action="store_true",
        help="Force a full re-scan of asset metadata, ignoring the cache.",
    )
    parser.add_argument(
        "--source",
        choices=["alpaca", "yfinance"],
        default="alpaca",
        help="Data provider to use.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="YYYY-MM-DD. Force ingestion to stop at this date.",
    )
    args = parser.parse_args()

    # 2. Setup Infrastructure
    log_manager = LogManager(service_name="ingest-worker")
    logger = log_manager.get_logger("main")

    logger.info(
        f"Initializing Ingest Worker | Mode: {args.mode} | Source: {args.source}"
    )

    try:
        # 3. Instantiate Components
        source = get_data_source(args.source)

        # Rate Limiter: YFinance is lenient, Alpaca is strict.
        # We use the config default (190/min) which is safe for both.
        limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
        clock = MarketClock()
        engine_logger = log_manager.get_logger("ingestion-engine")
        engine = IngestionEngine(
            source=source, logger=engine_logger, limiter=limiter, clock=clock
        )

        # 4. Execution Flow

        # Phase A: Asset Discovery
        # YFinance cannot scan for new assets, so we only sync metadata if using Alpaca.
        # Otherwise, we just load what we have.
        if args.source == "alpaca" and args.mode in ["all", "metadata"]:
            await engine.run_metadata_sync(force_rescan=args.force_metadata)
        else:
            await engine.load_existing_assets()

        # Stop here if we only wanted metadata
        if args.mode == "metadata":
            logger.info("Metadata sync complete. Exiting.")
            return

        # Phase B: Daily Data
        if args.mode in ["all", "daily"]:
            end_date_obj = None
            if args.end_date:
                try:
                    # Parse string to datetime
                    dt = datetime.strptime(args.end_date, "%Y-%m-%d")
                    # Convert to UTC using your helper
                    end_date_obj = ensure_utc_timestamp(dt)
                except ValueError:
                    logger.error("Invalid date format. Use YYYY-MM-DD.")
                    return
            await engine.run_daily_ingestion(end_date_override=end_date_obj)

        # Phase C: Intraday Data
        if args.mode in ["all", "intraday"]:
            if args.source == "yfinance":
                logger.warning(
                    "YFinance does not support deep intraday history. Skipping."
                )
            else:
                await engine.run_intraday_ingestion()

    except Exception as e:
        logger.exception("CRITICAL FAILURE in Ingest Worker")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
