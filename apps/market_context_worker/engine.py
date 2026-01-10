import asyncio
import polars as pl
from sqlalchemy import text, select, func
from datetime import datetime, timedelta
from typing import Dict

from packages.quant_lib.config import settings
from packages.database.session import get_db_session
from packages.database.models import Asset, MarketContextDaily, FeaturesDaily
from packages.quant_lib.market_context import MarketContextCalculator
from packages.quant_lib.market_clock import MarketClock


class MarketContextEngine:
    """
    Orchestrates the calculation and storage of daily, asset-agnostic market context indicators.
    Designed for a fast initial bulk load and even faster daily incremental updates.
    """

    REQUIRED_ASSETS = ["VIX", "SPY", "HYG", "IEF", "TLT"]

    def __init__(self, logger):
        self.logger = logger
        self.calculator = MarketContextCalculator()
        self.clock = MarketClock()
        self.db_url = (
            f"postgresql://{settings.db.user}:{settings.db.password}@"
            f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
        )

    async def run(
        self, force_full: bool = False, chunk_size_days: int = 90, concurrency: int = 4
    ):
        self.logger.info("--- Starting Market Context Worker ---")

        start_date = await self._get_next_processing_date(force_full)
        if start_date is None:
            self.logger.info("Market context is already up to date.")
            return

        self.logger.info(
            f"Processing market context from {start_date.date()} onwards..."
        )

        # 1. Load Asset Features ONCE (Small enough to keep in memory)
        # SPY/VIX/Rates for 20 years is only ~5000 rows each.
        self.logger.info("Pre-loading asset-specific context features...")
        asset_data_map = await self._load_required_asset_features(start_date)

        # Pre-calculate asset-based context (Trend, Credit, etc.) sans Breadth
        asset_context = self.calculator.calculate_asset_only_context(asset_data_map)

        # 2. Process Breadth in Chunks AND Upsert Immediately (Streaming)
        if force_full:
            await self._process_history_in_chunks(
                start_date, chunk_size_days, concurrency, asset_context
            )
        else:
            # Daily incremental logic
            breadth_df = await self._calculate_breadth_in_db(start_date)
            if not breadth_df.is_empty():
                final_df = self._join_and_finalize(breadth_df, asset_context)
                await self._write_context_to_db(final_df)

        self.logger.success("✅ Market Context update complete.")

    async def _get_next_processing_date(self, force_full: bool) -> datetime | None:
        async with get_db_session() as session:
            if force_full:
                self.logger.warning("FORCE MODE: Recalculating all market context.")
                result = await session.execute(select(func.min(FeaturesDaily.time)))
                min_date = result.scalar()
                # If DB is empty, default to something reasonable
                return min_date if min_date else datetime(2000, 1, 1)

            result = await session.execute(select(func.max(MarketContextDaily.time)))
            last_processed_date = result.scalar()

            if not last_processed_date:
                result = await session.execute(select(func.min(FeaturesDaily.time)))
                return result.scalar()

            last_settled = self.clock.get_last_settled_date()
            # Ensure types match for comparison
            if last_processed_date.date() >= last_settled:
                return None

            return last_processed_date + timedelta(days=1)

    async def _process_history_in_chunks(
        self,
        start_date: datetime,
        chunk_size_days: int,
        concurrency: int,
        asset_context: pl.DataFrame,
    ):
        self.logger.info(
            f"Processing history in {chunk_size_days}-day chunks with concurrency={concurrency}..."
        )

        async with get_db_session() as session:
            max_date_res = await session.execute(select(func.max(FeaturesDaily.time)))
            end_date = max_date_res.scalar()

        if not end_date:
            return

        date_chunks = []
        current_start = start_date
        while current_start <= end_date:
            current_end = current_start + timedelta(days=chunk_size_days - 1)
            date_chunks.append((current_start, min(current_end, end_date)))
            current_start = current_end + timedelta(days=1)

        self.logger.info(f"Scheduled {len(date_chunks)} chunks.")

        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        for i, (start, end) in enumerate(date_chunks):
            # Pass asset_context to the worker so it can join and write immediately
            task = asyncio.create_task(
                self._process_and_write_chunk(
                    i + 1, len(date_chunks), start, end, semaphore, asset_context
                )
            )
            tasks.append(task)

        # Wait for all chunks to finish writing
        await asyncio.gather(*tasks)

    async def _process_and_write_chunk(
        self, chunk_num, total, start, end, sem, asset_context
    ):
        """Worker that calculates breadth, joins context, and WRITES TO DB immediately."""
        async with sem:
            self.logger.info(
                f"  -> Processing Chunk {chunk_num}/{total} ({start.date()} to {end.date()})..."
            )

            try:
                # 1. Calc Breadth (Heavy DB query / Polars agg)
                breadth_df = await self._calculate_breadth_in_db(start, end)
                if breadth_df.is_empty():
                    return

                # 2. Join with Asset Context (In-Memory Polars Join)
                # Filter asset_context to just this chunk's timeframe to speed up join
                if asset_context is not None:
                    # Polars filtering requires timezone awareness matching
                    # Ensure start/end are timezone-aware if the DF is
                    chunk_assets = asset_context.filter(
                        (pl.col("time") >= start) & (pl.col("time") <= end)
                    )
                else:
                    chunk_assets = None

                final_df = self._join_and_finalize(breadth_df, chunk_assets)

                # 3. Write to DB Immediately
                rows = await self._write_context_to_db(final_df)
                self.logger.info(
                    f"     -> Chunk {chunk_num} Complete: Wrote {rows} rows."
                )

            except Exception as e:
                self.logger.error(f"Failed chunk {chunk_num}: {e}")

    async def _calculate_breadth_in_db(
        self, start_date: datetime, end_date: datetime = None
    ) -> pl.DataFrame:
        """
        Calculates breadth stats using a Polars aggregation on fetched columns.
        This shifts load from DB CPU to App CPU (M1 Max).
        """
        # Fetch only necessary columns to minimize I/O
        query = f"""
            SELECT fd.time, mdd.close, fd.sma_20, fd.sma_50, fd.sma_200
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            WHERE fd.time >= '{start_date.isoformat()}'
        """
        if end_date:
            query += f" AND fd.time <= '{end_date.isoformat()}'"

        # Use asyncio.to_thread for blocking IO
        df = await asyncio.to_thread(pl.read_database_uri, query, self.db_url)

        if df.is_empty():
            return pl.DataFrame()

        # Aggregate in Polars (Fast)
        return (
            df.group_by("time")
            .agg(
                [
                    (pl.col("close") > pl.col("sma_20"))
                    .mean()
                    .alias("breadth_pct_above_sma20"),
                    (pl.col("close") > pl.col("sma_50"))
                    .mean()
                    .alias("breadth_pct_above_sma50"),
                    (pl.col("close") > pl.col("sma_200"))
                    .mean()
                    .alias("breadth_pct_above_sma200"),
                ]
            )
            .sort("time")
        )

    def _join_and_finalize(
        self, breadth_df: pl.DataFrame, asset_context: pl.DataFrame
    ) -> pl.DataFrame:
        if asset_context is not None and not asset_context.is_empty():
            return (
                breadth_df.join(asset_context, on="time", how="left")
                .sort("time")
                .fill_null(strategy="forward")
            )

        return breadth_df

    async def _load_required_asset_features(
        self, start_date: datetime
    ) -> Dict[str, pl.DataFrame]:
        tasks = {
            asset: self._load_features_for_asset(asset, start_date)
            for asset in self.REQUIRED_ASSETS
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    async def _load_features_for_asset(
        self, symbol: str, start_date: datetime
    ) -> pl.DataFrame:
        """
        OPTIMIZED: Fetches Features and Prices separately to avoid DB-side Joins.
        Joins in memory on the Client (Mac) to reduce NAS I/O pressure.
        """
        # 1. Resolve Asset ID first (Lightweight)
        async with get_db_session() as session:
            result = await session.execute(
                select(Asset.id).where(Asset.symbol == symbol)
            )
            asset_id = result.scalar()

        if not asset_id:
            self.logger.warning(f"Asset '{symbol}' not found in DB.")
            return pl.DataFrame()

        # 2. Construct Simple Queries (No Joins = Fast Index Scans)
        # Fetch all columns from features_daily
        q_features = f"""
            SELECT * 
            FROM features_daily 
            WHERE asset_id = {asset_id} AND time >= '{start_date.isoformat()}'
            ORDER BY time ASC
        """

        # Fetch just close price from market_data_daily
        q_prices = f"""
            SELECT time, close 
            FROM market_data_daily 
            WHERE asset_id = {asset_id} AND time >= '{start_date.isoformat()}'
            ORDER BY time ASC
        """

        # 3. Execute in Parallel (Low IO overhead compared to Join)
        self.logger.info(f"  -> Fetching data for {symbol}...")
        features_df, prices_df = await asyncio.gather(
            asyncio.to_thread(pl.read_database_uri, q_features, self.db_url),
            asyncio.to_thread(pl.read_database_uri, q_prices, self.db_url),
        )

        if features_df.is_empty():
            return pl.DataFrame()

        # 4. Join in Memory (Instant on M1)
        # We join on 'time'. Note: features_daily already has asset_id, prices_df implies it.
        return features_df.join(prices_df, on="time", how="left")

    async def _write_context_to_db(self, df: pl.DataFrame):
        """High-Performance Upsert for MarketContextDaily with progress logging."""

        if df.is_empty():
            return

        total_rows = len(df)
        self.logger.info(f"Preparing to upsert {total_rows} rows to the database...")

        cols = ["time"] + [c for c in df.columns if c != "time"]
        # Cast to float32 to save network bandwidth and DB space if desired, but float64 is safer standard
        df_clean = df.select(cols).fill_nan(None)
        records = list(df_clean.iter_rows())

        async with get_db_session() as session:
            conn = await session.connection()
            dbapi_conn = await conn.get_raw_connection()
            asyncpg_conn = dbapi_conn.driver_connection

            temp_table, target_table = (
                "temp_context_upsert",
                MarketContextDaily.__tablename__,
            )
            await session.execute(
                text(
                    f'CREATE TEMP TABLE {temp_table} (LIKE "{target_table}") ON COMMIT DROP;'
                )
            )

            self.logger.info(f"Copying {total_rows} rows to temporary table...")
            await asyncpg_conn.copy_records_to_table(
                temp_table, records=records, columns=cols
            )

            update_cols = [c for c in cols if c != "time"]
            update_set_str = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])
            cols_str = ", ".join([f'"{c}"' for c in cols])

            upsert_query = f"""
                INSERT INTO "{target_table}" ({cols_str}) SELECT {cols_str} FROM {temp_table}
                ON CONFLICT (time) DO UPDATE SET {update_set_str};
            """

            self.logger.info("Executing final upsert from temp table...")
            await session.execute(text(upsert_query))
            await session.commit()
            return len(records)
