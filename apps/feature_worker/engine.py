# apps/feature_worker/engine.py

import asyncio
from datetime import datetime, timedelta, timezone
import polars as pl
from sqlalchemy import func, select, text

from packages.quant_lib.config import settings
from packages.quant_lib.features import FeatureFactory
from packages.database.session import get_db_session
from packages.database.models import Asset, FeaturesDaily


class FeatureEngine:
    def __init__(self, logger):
        self.logger = logger
        self.factory = FeatureFactory(settings)
        self.active_assets_map: dict[int, str] = {}
        self.concurrency_limiter = asyncio.Semaphore(5)

        # Calculation: 252 trading days * 1.4 (weekends) + buffer = ~400.
        # We use 500 to be extremely safe for EMA convergence.
        self.LOOKBACK_BUFFER_DAYS = 500

        self.processed_count = 0
        self.total_assets = 0
        self._counter_lock = asyncio.Lock()  # Use asyncio.Lock

    async def run(self, force_full: bool = False, symbol: str | None = None):
        """
        Main entry point.
        :param force_full: If True, ignores incremental logic and recalculates history.
        :param symbol: If set, only processes this specific ticker.
        """
        await self._load_active_assets(specific_symbol=symbol)

        if not self.active_assets_map:
            return

        self.total_assets = len(self.active_assets_map)
        self.processed_count = 0

        # Load benchmark data (SPY) once for relative calculations across all time
        self.logger.info("Loading Benchmark (SPY) data...")
        benchmark_df = await self._load_market_data(
            symbol="SPY", asset_id=None, min_date=None
        )
        tasks = []

        if force_full:
            self.logger.warning(
                "⚠️ FORCE MODE ENABLED: Ignoring incremental logic. This will re-process ALL history."
            )

        for asset_id, symbol in self.active_assets_map.items():
            # Spawn a task for each asset
            task = asyncio.create_task(
                self._process_single_asset(asset_id, symbol, benchmark_df, force_full)
            )
            tasks.append(task)

        # Wait for all tasks to finish
        # The Semaphore inside _process_single_asset prevents us from running 3200 at once
        if tasks:
            await asyncio.gather(*tasks)

        self.logger.success("Feature generation complete for all active assets.")

        if force_full:
            self.logger.info("Re-compressing data...")
            await self._manage_compression(disable=False)

    async def _load_active_assets(self, specific_symbol: str | None = None):
        """
        Loads the map of assets marked as active from the database.
        If specific_symbol is provided, filters for just that one.
        """
        async with get_db_session() as session:
            stmt = select(Asset.id, Asset.symbol).where(Asset.is_active == True)

            if specific_symbol:
                stmt = stmt.where(Asset.symbol == specific_symbol.upper())

            stmt = stmt.order_by(Asset.symbol)

            result = await session.execute(stmt)

            self.active_assets_map = {row[0]: row[1] for row in result}

        if specific_symbol and not self.active_assets_map:
            self.logger.warning(f"Symbol '{specific_symbol}' not found or not active.")
        else:
            self.logger.info(
                f"Loaded {len(self.active_assets_map)} active assets to process (Sorted by Symbol)."
            )

    async def _process_single_asset(
        self, asset_id: int, symbol: str, benchmark_df: pl.DataFrame, force_full: bool
    ):
        """
        Processes a single asset.
        Protected by a Semaphore to prevent exploding RAM/DB connections.
        """
        async with self.concurrency_limiter:
            # 1. Check Last Feature Date. Force full history if forced
            last_feature_time = None
            if not force_full:
                last_feature_time = await self._get_last_feature_date(asset_id)

            # 2. Determine Load Start Date
            load_start_date = None
            is_incremental = False

            if last_feature_time:
                # Incremental Mode: Load from (Last - Buffer)
                load_start_date = last_feature_time - timedelta(
                    days=self.LOOKBACK_BUFFER_DAYS
                )
                is_incremental = True
                # self.logger.debug(f"{symbol}: Incremental update from {last_feature_time.date()}")
            else:
                # Full Mode: Load everything
                # self.logger.debug(f"{symbol}: Full calculation (No history found)")
                pass

            # 3. Load Data
            asset_df = await self._load_market_data(
                symbol, asset_id, min_date=load_start_date
            )

            if asset_df.is_empty():
                return

            # 4. Generate Features
            # We calculate features on the whole buffer to ensure correctness
            try:
                features_df = self.factory.generate_all(asset_df, benchmark_df)
            except Exception as e:
                self.logger.error(f"Failed to generate features for {symbol}: {e}")
                return

            self.logger.info(f"Processing {symbol} (ID: {asset_id})...")

            # 5. Filter Output (Optimization)
            # If incremental, we discard the "Buffer" rows (we only re-calculated them for math context)
            # We only write rows NEWER than what we already have in DB.
            if is_incremental and last_feature_time:
                # Polars filtering
                # Ensure last_feature_time is timezone aware for comparison
                cutoff = last_feature_time.replace(tzinfo=timezone.utc)
                features_df = features_df.filter(pl.col("time") > cutoff)

            if features_df.is_empty():
                # self.logger.debug(f"{symbol}: No new features to write.")
                return

            # 6. Write to DB
            rows_written = await self._write_features_to_db(features_df)

            async with self._counter_lock:
                self.processed_count += 1

            # Only log every 25 assets
            if self.processed_count % 25 == 0:
                progress_pct = (self.processed_count / self.total_assets) * 100
                self.logger.info(
                    f"Progress: {self.processed_count}/{self.total_assets} ({progress_pct:.1f}%) | Last: Wrote {rows_written} for {symbol}."
                )

    async def _get_last_feature_date(self, asset_id: int) -> datetime | None:
        """Finds the timestamp of the most recent entry in features_daily."""
        async with get_db_session() as session:
            result = await session.execute(
                select(func.max(FeaturesDaily.time)).where(
                    FeaturesDaily.asset_id == asset_id
                )
            )
            return result.scalar()

    async def _load_market_data(
        self, symbol: str, asset_id: int = None, min_date: datetime = None
    ) -> pl.DataFrame:
        """
        Loads daily market data.
        If min_date is provided, only loads data after that date.
        """
        if asset_id is None:
            async with get_db_session() as session:
                result = await session.execute(
                    select(Asset.id).where(Asset.symbol == symbol)
                )
                asset_id = result.scalar_one_or_none()

        if not asset_id:
            return pl.DataFrame()

        db_url = (
            f"postgresql://{settings.db.user}:{settings.db.password}@"
            f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
        )

        query = f"SELECT * FROM market_data_daily WHERE asset_id = {asset_id}"
        if min_date:
            # Format datetime for SQL
            query += f" AND time >= '{min_date.isoformat()}'"

        query += " ORDER BY time ASC"

        try:
            # Use connectorx
            df = pl.read_database_uri(query=query, uri=db_url)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol}: {e}")
            return pl.DataFrame()

    async def _write_features_to_db(self, df: pl.DataFrame):
        """
        High-Performance Upsert using a Temp Table + COPY.
        """
        if df.is_empty():
            return

        # 1. Prepare Columns
        model_columns = {c.name for c in FeaturesDaily.__table__.columns}
        df_columns = set(df.columns)
        valid_cols = list(model_columns.intersection(df_columns))

        clean_exprs = []
        for col_name in valid_cols:
            dtype = df[col_name].dtype

            # Floats can be NaN, so we clean them
            if dtype in (pl.Float64, pl.Float32):
                clean_exprs.append(pl.col(col_name).fill_nan(None))
            else:
                # Pass other types (time, asset_id, etc.) through unchanged
                clean_exprs.append(pl.col(col_name))

        # Convert NaNs to Nulls
        df_clean = df.select(clean_exprs)

        # Prepare for COPY
        records = list(df_clean.select(valid_cols).iter_rows())

        if not records:
            return

        # 2. Staging & COPY
        async with get_db_session() as session:
            # Get raw connection
            conn = await session.connection()
            dbapi_conn = await conn.get_raw_connection()
            asyncpg_conn = dbapi_conn.driver_connection

            # Create Temp Table
            temp_table = "temp_features_upsert"
            target_table = FeaturesDaily.__tablename__

            await session.execute(
                text(
                    f'CREATE TEMP TABLE {temp_table} (LIKE "{target_table}") ON COMMIT DROP;'
                )
            )

            # COPY data into it (Fast)
            await asyncpg_conn.copy_records_to_table(
                temp_table, records=records, columns=valid_cols
            )

            # 3. UPSERT from Temp to Real
            # This is the "INSERT ... ON CONFLICT DO UPDATE" command

            # List of columns for INSERT
            cols_str = ", ".join([f'"{c}"' for c in valid_cols])

            # List of columns to UPDATE
            # (All columns except the primary keys 'time', 'asset_id')
            update_cols = [c for c in valid_cols if c not in ["time", "asset_id"]]
            update_set_str = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])

            upsert_query = f"""
                INSERT INTO "{target_table}" ({cols_str})
                SELECT {cols_str} FROM {temp_table}
                ON CONFLICT (time, asset_id) DO UPDATE
                SET {update_set_str};
            """

            await session.execute(text(upsert_query))
            await session.commit()

        # self.logger.success(f"Successfully upserted {len(records)} feature rows.")
        return len(records)

    async def _manage_compression(self, disable: bool):
        """
        Manages TimescaleDB compression for the features table.
        disable=True: Decompresses ALL chunks.
        disable=False: Triggers the background policy job to re-compress.
        """
        table_name = FeaturesDaily.__tablename__

        async with get_db_session() as session:
            if disable:
                self.logger.warning(
                    f"Preparing for --force: Decompressing '{table_name}' chunks..."
                )

                # This query finds all compressed chunks for the table and decompresses them.
                query = text(
                    f"""
                    SELECT decompress_chunk(c.chunk_schema || '.' || c.chunk_name)
                    FROM timescaledb_information.chunks c
                    WHERE c.hypertable_name = '{table_name}' AND c.is_compressed = TRUE;
                """
                )

                try:
                    await session.execute(query)
                    await session.commit()
                    self.logger.success("Decompression complete.")
                except Exception as e:
                    self.logger.error(f"Failed to decompress chunks: {e}")
            else:
                # Re-enable / Run Policy
                self.logger.info(
                    f"Triggering background compression for '{table_name}'..."
                )

                # This command forces the policy job to run now instead of waiting for its schedule.
                query = text(
                    f"""
                    SELECT run_job(job_id)
                    FROM timescaledb_information.jobs
                    WHERE proc_name = 'policy_compression' AND hypertable_name = '{table_name}';
                """
                )

                try:
                    await session.execute(query)
                    await session.commit()
                    self.logger.success("Compression job triggered.")
                except Exception as e:
                    self.logger.warning(f"Could not trigger compression job: {e}")
