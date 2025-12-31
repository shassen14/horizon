# apps/feature_worker/engine.py

import asyncio
from datetime import datetime, timedelta, timezone
import polars as pl
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

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

    async def run(self, force_full: bool = False, symbol: str | None = None):
        """
        Main entry point.
        :param force_full: If True, ignores incremental logic and recalculates history.
        :param symbol: If set, only processes this specific ticker.
        """
        await self._load_active_assets(specific_symbol=symbol)

        if not self.active_assets_map:
            return

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
            await self._write_features_to_db(features_df)

            self.logger.info(f"{symbol}: Wrote {len(features_df)} new rows.")

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
        """Upserts a DataFrame of features into the features_daily table."""
        if df.is_empty():
            return

        # 1. Identify valid columns
        model_columns = {c.name for c in FeaturesDaily.__table__.columns}
        df_columns = set(df.columns)
        columns_to_insert = list(model_columns.intersection(df_columns))

        # Polars treats NaN and Null differently. We want to convert NaNs to Nulls.
        # This ensures Postgres gets NULL, which it handles correctly in aggregates.
        exprs = []
        for col_name in columns_to_insert:
            dtype = df[col_name].dtype

            # Only Floats support NaN.
            # Integers (asset_id) and Datetimes (time) throws error if we check for NaN.
            if dtype in (pl.Float64, pl.Float32):
                exprs.append(pl.col(col_name).fill_nan(None))
            else:
                # Pass other columns (time, asset_id) through unchanged
                exprs.append(pl.col(col_name))

        df_clean = df.select(exprs)

        # Prepare records
        records = df_clean.to_dicts()

        if not records:
            return

        # 2. Dynamic Chunk Sizing
        # POSTGRES LIMIT: 32,767 (Signed 16-bit integer).
        # We use 30,000 to provide a safety buffer.
        POSTGRES_PARAM_LIMIT = 30_000
        num_columns = len(columns_to_insert)

        # Calculate safe chunk size: 30,000 / 30 cols = 1,000 rows
        safe_chunk_size = POSTGRES_PARAM_LIMIT // num_columns

        # Clamp it to something reasonable (e.g., at least 1, max 10k)
        safe_chunk_size = max(1, min(10_000, safe_chunk_size))

        self.logger.debug(
            f"Dynamic Write: {num_columns} cols -> Chunk size {safe_chunk_size}"
        )

        async with get_db_session() as session:
            # Upsert Statement Setup
            stmt = insert(FeaturesDaily).values(records)
            update_dict = {
                col.name: col
                for col in stmt.excluded
                if col.name not in ["time", "asset_id"]
            }

            # 3. Execution Loop
            total_records = len(records)
            for i in range(0, total_records, safe_chunk_size):
                chunk = records[i : i + safe_chunk_size]

                # We reconstruct the insert statement for the chunk
                # (SQLAlchemy insert objects are reusable but values binding is cleaner this way)
                stmt = insert(FeaturesDaily).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["time", "asset_id"],
                    set_=update_dict,
                )

                await session.execute(stmt)
                await session.commit()

        self.logger.success(
            f"Successfully upserted {len(records)} feature rows (Chunk size: {safe_chunk_size})."
        )
