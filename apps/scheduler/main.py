import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import utc

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager

# Import Engines
from apps.ingest_worker.engine import IngestionEngine
from apps.ingest_worker.sources.alpaca import AlpacaSource
from apps.feature_worker.engine import FeatureEngine
from aiolimiter import AsyncLimiter

# Initialize Logger
log_manager = LogManager(settings, "scheduler")
logger = log_manager.get_logger("main")


async def job_ingest_intraday():
    logger.info("⏰ TRIGGER: Starting Intraday Ingestion...")
    try:
        # Re-initialize dependencies per job to ensure clean state
        source = AlpacaSource()
        limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
        engine_logger = log_manager.get_logger("ingest-intraday")

        engine = IngestionEngine(source, settings, engine_logger, limiter)
        await engine.run_intraday_ingestion()
    except Exception as e:
        logger.exception(f"Intraday Job Failed: {e}")


async def job_daily_routine():
    logger.info("⏰ TRIGGER: Starting Daily Routine (Metadata -> Daily -> Features)...")
    try:
        # 1. Ingestion
        source = AlpacaSource()
        limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
        ingest_logger = log_manager.get_logger("ingest-daily")
        ingest_engine = IngestionEngine(source, settings, ingest_logger, limiter)

        # Run Metadata & Daily
        await ingest_engine.run_metadata_sync()
        await ingest_engine.run_daily_ingestion()

        # 2. Features (Runs immediately after ingestion finishes)
        feat_logger = log_manager.get_logger("feature-daily")
        feature_engine = FeatureEngine(settings, feat_logger)
        await feature_engine.run()

        logger.success("✅ Daily Routine Complete.")

    except Exception as e:
        logger.exception(f"Daily Routine Failed: {e}")


async def main():
    scheduler = AsyncIOScheduler(timezone=utc)

    # --- Schedule Jobs ---

    # 1. Intraday: Every 15 minutes during market hours (simplistic cron for now)
    # Ideally, you'd range this 9am-5pm EST
    scheduler.add_job(
        job_ingest_intraday, CronTrigger(minute="*/15", day_of_week="mon-fri")
    )

    # 2. Daily Routine: Runs at 5:00 PM EST (22:00 UTC approx)
    # Adjust hour based on when Alpaca SIP data settles
    scheduler.add_job(
        job_daily_routine, CronTrigger(hour=22, minute=15, day_of_week="mon-fri")
    )

    logger.info("--- Horizon Scheduler Started ---")
    scheduler.start()

    # Keep the process alive
    try:
        while True:
            await asyncio.sleep(1000)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    asyncio.run(main())
