# apps/db_worker/policies.py

from sqlalchemy import text
from packages.database.models import MarketDataDaily, get_model_for_timeframe
from packages.database.session import get_db_session
from packages.quant_lib.config import settings


class PolicyManager:
    def __init__(self, logger):
        self.logger = logger

    async def apply_all(self):
        self.logger.info("--- Applying Database Policies ---")
        await self._apply_daily_policies()
        await self._apply_intraday_policies()

    async def _apply_daily_policies(self):
        # Retention: Keep Daily data forever (no policy needed)
        # Compression: Compress data older than 30 days
        table_name = MarketDataDaily.__tablename__

        await self._enable_compression(
            table=table_name,
            compress_after="30 days",
            segment_by="asset_id",
            order_by="time DESC",
        )

    async def _apply_intraday_policies(self):
        # Retention: Drop 5-min data older than Config (e.g. 3 Years)
        # We add a small buffer (e.g. +1 month) to the config value
        days = settings.ingestion.intraday_lookback_days + 30

        IntradayModel = get_model_for_timeframe(
            settings.ingestion.intraday_timeframe_unit,
            settings.ingestion.intraday_timeframe_value,
        )
        table_name = IntradayModel.__tablename__

        await self._add_retention_policy(table_name, f"{days} days")

        # Compression: Compress data older than 7 days
        # This keeps the "Hot" data (last 7 days) uncompressed for fast inserts/updates,
        # and compresses the "Cold" history.
        await self._enable_compression(
            table=table_name,
            compress_after="7 days",
            segment_by="asset_id",
            order_by="time DESC",
        )

    async def _add_retention_policy(self, table: str, interval: str):
        async with get_db_session() as session:
            # Check if policy exists
            exists_query = text(
                f"""
                SELECT count(*) FROM timescaledb_information.jobs j
                WHERE j.proc_name = 'policy_retention' 
                AND j.hypertable_name = '{table}';
            """
            )
            result = await session.execute(exists_query)

            # If exists, REMOVE it first (to apply the new setting)
            if result.scalar() > 0:
                self.logger.info(f"Removing old retention policy for {table}...")
                try:
                    await session.execute(
                        text(f"SELECT remove_retention_policy('{table}');")
                    )
                except Exception as e:
                    self.logger.warning(f"Could not remove old policy: {e}")

            # Add the NEW policy
            self.logger.info(
                f"Adding Retention Policy to {table} (Drop after {interval})"
            )
            await session.execute(
                text(f"SELECT add_retention_policy('{table}', INTERVAL '{interval}');")
            )

    async def _enable_compression(
        self, table: str, compress_after: str, segment_by: str, order_by: str
    ):
        async with get_db_session() as session:
            # Check if compression is ALREADY enabled
            # We query the timescaledb metadata view
            check_query = text(
                f"""
                SELECT compression_enabled 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name = '{table}';
            """
            )
            result = await session.execute(check_query)
            is_enabled = result.scalar()

            if not is_enabled:
                self.logger.info(f"Enabling Compression on {table}...")
                await session.execute(
                    text(
                        f"""
                    ALTER TABLE "{table}" SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = '{segment_by}',
                        timescaledb.compress_orderby = '{order_by}'
                    );
                """
                    )
                )
            else:
                self.logger.info(f"Compression already enabled on {table}.")

            # Handle the Automation Policy (Add/Update)
            # Check if policy exists
            policy_exists_query = text(
                f"""
                SELECT count(*) FROM timescaledb_information.jobs j
                WHERE j.proc_name = 'policy_compression' 
                AND j.hypertable_name = '{table}';
            """
            )
            result = await session.execute(policy_exists_query)

            if result.scalar() > 0:
                # Remove old policy to ensure we apply the new 'compress_after' setting
                # (e.g. changing from 30 days to 7 days)
                self.logger.info(f"Refreshing compression policy for {table}...")
                try:
                    await session.execute(
                        text(f"SELECT remove_compression_policy('{table}');")
                    )
                except Exception:
                    pass

            self.logger.info(
                f"Adding Compression Policy to {table} (Compress after {compress_after})"
            )
            await session.execute(
                text(
                    f"SELECT add_compression_policy('{table}', INTERVAL '{compress_after}');"
                )
            )
