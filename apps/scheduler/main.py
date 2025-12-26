# apps/scheduler/main.py

import asyncio
from datetime import datetime, timezone

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from market_clock import MarketClock

# Import Engines
from apps.ingest_worker.engine import IngestionEngine
from apps.ingest_worker.sources.alpaca import AlpacaSource
from apps.feature_worker.engine import FeatureEngine
from aiolimiter import AsyncLimiter

# Initialize Logger
log_manager = LogManager(settings, "scheduler")
logger = log_manager.get_logger("main")

# --- Job Locks ---
# These prevent a job from being triggered if the previous run is still active
ingest_intraday_lock = asyncio.Lock()
daily_routine_lock = asyncio.Lock()


# --- Job Functions (Same as before, but now with logging) ---
async def job_ingest_intraday():
    if ingest_intraday_lock.locked():
        logger.warning("Intraday job skipped: Previous run still active.")
        return

    async with ingest_intraday_lock:
        logger.info(">>> Starting Intraday Ingestion...")
        try:
            source = AlpacaSource()
            limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
            engine_logger = log_manager.get_logger("ingest-intraday")
            engine = IngestionEngine(source, settings, engine_logger, limiter)

            # await engine.run_metadata_sync()
            await engine.run_intraday_ingestion()
            logger.success("<<< Intraday Ingestion Complete.")
        except Exception as e:
            logger.exception(f"Intraday Job Failed: {e}")


async def job_daily_routine():
    if daily_routine_lock.locked():
        logger.warning("Daily routine skipped: Previous run still active.")
        return

    async with daily_routine_lock:
        logger.info(">>> Starting Daily Routine...")
        try:
            # 1. Ingestion
            source = AlpacaSource()
            limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
            ingest_logger = log_manager.get_logger("ingest-daily")
            ingest_engine = IngestionEngine(source, settings, ingest_logger, limiter)

            await ingest_engine.run_metadata_sync()
            await ingest_engine.run_daily_ingestion()

            # 2. Features
            feat_logger = log_manager.get_logger("feature-daily")
            feature_engine = FeatureEngine(settings, feat_logger)
            await feature_engine.run()

            logger.success("<<< Daily Routine Complete.")
        except Exception as e:
            logger.exception(f"Daily Routine Failed: {e}")


# --- The Main Loop ---
async def main():
    clock = MarketClock()

    # Track state to run daily jobs only once
    last_daily_run_date = None

    logger.info("--- Horizon Scheduler Started: Market-Aware Mode ---")

    while True:
        now = datetime.now(timezone.utc)

        # --- Intraday Logic ---
        # Run every 5 minutes during market hours
        if clock.is_market_open():
            if now.minute % 5 == 0:
                logger.info(
                    f"Market is OPEN. Minute is {now.minute}. Triggering Intraday job."
                )
                asyncio.create_task(job_ingest_intraday())

        # --- Daily Logic ---
        # Run once after market close
        # (e.g., 4:15 PM EST -> ~21:15 UTC)

        is_weekday = now.weekday() < 5  # Mon=0, Fri=4
        is_after_close = now.hour >= 21 and now.minute >= 15

        # Check if it's time AND we haven't run it today yet
        if is_weekday and is_after_close and now.date() != last_daily_run_date:
            logger.info(f"Market is likely CLOSED. Triggering Daily Routine.")
            asyncio.create_task(job_daily_routine())
            last_daily_run_date = now.date()  # Mark as run for today

        # Tick every minute
        await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down.")
