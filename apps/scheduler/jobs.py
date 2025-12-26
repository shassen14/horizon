# apps/scheduler/jobs.py

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from aiolimiter import AsyncLimiter

# --- Engine Imports ---
from apps.ingest_worker.engine import IngestionEngine
from apps.ingest_worker.sources.alpaca import AlpacaSource
from apps.feature_worker.engine import FeatureEngine
from apps.db_worker.auditor import DataAuditor
from apps.db_worker.optimization import DatabaseOptimizer

# from apps.db_worker.backup import BackupManager


class JobRunner:
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager

    async def run_ingest_intraday(self):
        logger = self.log_manager.get_logger("ingest-intraday")
        try:
            logger.info(">>> Starting...")
            source = AlpacaSource()
            limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
            engine = IngestionEngine(source, logger, limiter)

            await engine.run_metadata_sync()  # Fast cache check
            await engine.run_intraday_ingestion()
            logger.success("<<< Complete.")
        except Exception as e:
            logger.exception(f"Job Failed: {e}")

    async def run_daily_routine(self):
        logger = self.log_manager.get_logger("daily-routine")
        try:
            logger.info(">>> Starting Daily Ingestion & Feature Gen...")

            # 1. Ingestion
            source = AlpacaSource()
            limiter = AsyncLimiter(settings.ingestion.api_rate_limit_per_minute, 60)
            ingest_engine = IngestionEngine(source, logger, limiter)
            await ingest_engine.run_metadata_sync()
            await ingest_engine.run_daily_ingestion()

            # 2. Features
            feature_engine = FeatureEngine(logger)
            await feature_engine.run()

            logger.success("<<< Complete.")
        except Exception as e:
            logger.exception(f"Job Failed: {e}")

    async def run_data_audit(self):
        logger = self.log_manager.get_logger("data-audit")
        try:
            logger.info(">>> Starting Data Audit...")
            auditor = DataAuditor(logger)
            await auditor.run_daily_audit()
            await auditor.run_intraday_audit()
            logger.success("<<< Complete.")
        except Exception as e:
            logger.exception(f"Job Failed: {e}")

    async def run_db_optimize(self):
        logger = self.log_manager.get_logger("db-optimize")
        try:
            logger.info(">>> Starting DB Optimization...")
            optimizer = DatabaseOptimizer(logger)
            await optimizer.run_maintenance()
            logger.success("<<< Complete.")
        except Exception as e:
            logger.exception(f"Job Failed: {e}")

    # def run_db_backup(self):
    #     logger = self.log_manager.get_logger("db-backup")
    #     try:
    #         logger.info(">>> Starting DB Backup...")
    #         # BackupManager uses subprocess, which can be blocking.
    #         # Running this in a separate thread prevents blocking the main scheduler loop.
    #         backup_manager = BackupManager(logger)
    #         backup_manager.run_backup()
    #         logger.success("<<< Complete.")
    #     except Exception as e:
    #         logger.exception(f"Job Failed: {e}")
