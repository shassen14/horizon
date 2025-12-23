# apps/db_worker/optimization.py

from sqlalchemy import text
from packages.database.session import get_autocommit_connection
from packages.database.models import (
    Asset,
    MarketDataDaily,
    FeaturesDaily,
    get_model_for_timeframe,
)
from packages.quant_lib.config import settings


class DatabaseOptimizer:
    def __init__(self, logger):
        self.logger = logger

    async def run_maintenance(self):
        self.logger.info("--- Running Database Maintenance (VACUUM ANALYZE) ---")

        # This ensures we optimize "market_data_5min" (or 1min, or 1hour)
        # depending on what is actually running.
        try:
            IntradayModel = get_model_for_timeframe(
                settings.ingestion.intraday_timeframe_unit,
                settings.ingestion.intraday_timeframe_value,
            )
        except ValueError:
            self.logger.error("Could not resolve Intraday model from config.")
            return

        # Build List of Target Models
        target_models = [Asset, MarketDataDaily, FeaturesDaily, IntradayModel]

        async with get_autocommit_connection() as conn:

            # Optimize Tables dynamically
            for model in target_models:
                table_name = model.__tablename__  # <--- The Source of Truth
                self.logger.info(f"Vacuuming {table_name}...")
                await conn.execute(text(f'VACUUM ANALYZE "{table_name}";'))

        self.logger.success("Maintenance complete.")
