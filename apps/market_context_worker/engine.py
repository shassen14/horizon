import asyncio
import polars as pl
from sqlalchemy import text, select, func
from datetime import datetime, date, timedelta
from typing import Dict

# --- Library Imports ---
from packages.quant_lib.config import settings
from packages.database.session import get_db_session
from packages.database.models import MarketContextDaily, FeaturesDaily
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
        self, force_full: bool = False, chunk_size_days: int = 90, concurrency: int = 2
    ):
        self.logger.info("--- Starting Market Context Worker ---")

        start_date = await self._get_next_processing_date(force_full)
        if start_date is None:
            self.logger.info("Market context is already up to date.")
            return

        self.logger.info(
            f"Processing market context from {start_date.date()} onwards..."
        )

        # --- 1. FETCH ALL RAW DATA ---
        self.logger.info("Fetching all required raw data sources...")

        # A. Fetch asset-specific features
        asset_tasks = {
            asset: self._load_features_for_asset(asset, start_date)
            for asset in self.REQUIRED_ASSETS
        }

        # B. Fetch breadth data (the heavy part)
        if force_full:
            breadth_task = self._calculate_breadth_in_chunks(
                start_date, chunk_size_days, concurrency
            )
        else:
            breadth_task = self._calculate_breadth_in_db(start_date)

        # C. Run all I/O tasks concurrently
        all_tasks = {**asset_tasks, "BREADTH_FEATURES": breadth_task}
        results = await asyncio.gather(*all_tasks.values())
        data_map = dict(zip(all_tasks.keys(), results))

        # --- 2. DELEGATE TO CALCULATOR ---
        self.logger.info("Calculating final context DataFrame...")
        # The calculator's `calculate_all` method now handles the joining logic
        # We pass the pre-calculated breadth features directly.
        final_df = self.calculator.calculate_all_from_sources(
            asset_data=data_map, breadth_features=data_map.get("BREADTH_FEATURES")
        )

        if final_df is None or final_df.is_empty():
            self.logger.warning(
                "Context calculation resulted in an empty DataFrame. Nothing to write."
            )
            return

        # --- 3. WRITE RESULTS ---
        await self._write_context_to_db(final_df)
        self.logger.success("✅ Market Context update complete.")

    async def _get_next_processing_date(self, force_full: bool) -> date | None:
        """Determines the start date for processing, enabling incremental updates."""
        async with get_db_session() as session:
            if force_full:
                self.logger.warning("FORCE MODE: Recalculating all market context.")
                result = await session.execute(select(func.min(FeaturesDaily.time)))
                return result.scalar()

            # Find the last processed date
            result = await session.execute(select(func.max(MarketContextDaily.time)))
            last_processed_date = result.scalar()

            if not last_processed_date:
                self.logger.info(
                    "No existing market context found. Starting from the beginning."
                )
                result = await session.execute(select(func.min(FeaturesDaily.time)))
                return result.scalar()

            # Check if we are already up to date
            last_settled_market_day = self.clock.get_last_settled_date()
            if last_processed_date.date() >= last_settled_market_day:
                return None  # We are current

            # Start processing from the day after the last successful run
            return last_processed_date.date() + timedelta(days=1)

    async def _calculate_breadth_in_chunks(
        self, start_date: date, chunk_size_days: int, concurrency: int
    ) -> pl.DataFrame:
        """Processes the entire history in parallel time-based chunks for the initial bulk load."""
        self.logger.info(
            f"Performing initial bulk breadth calculation in {chunk_size_days}-day chunks..."
        )

        async with get_db_session() as session:
            max_date_res = await session.execute(select(func.max(FeaturesDaily.time)))
            end_date = max_date_res.scalar()

        if not end_date:
            return pl.DataFrame()

        date_chunks = []
        current_start = start_date
        while current_start <= end_date:
            current_end = current_start + timedelta(days=chunk_size_days - 1)
            date_chunks.append((current_start, min(current_end, end_date)))
            current_start = current_end + timedelta(days=1)

        self.logger.info(
            f"Divided history into {len(date_chunks)} chunks. Processing with concurrency={concurrency}..."
        )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._process_breadth_chunk(i + 1, len(date_chunks), start, end, semaphore)
            for i, (start, end) in enumerate(date_chunks)
        ]

        chunk_results = await asyncio.gather(*tasks)
        return pl.concat(
            [df for df in chunk_results if df is not None and not df.is_empty()]
        )

    async def _process_breadth_chunk(
        self,
        chunk_num: int,
        total_chunks: int,
        start: date,
        end: date,
        sem: asyncio.Semaphore,
    ) -> pl.DataFrame | None:
        """Worker function to process a single time chunk for breadth with progress logging."""
        async with sem:
            progress = (chunk_num / total_chunks) * 100
            self.logger.info(
                f"  -> Processing chunk {chunk_num}/{total_chunks} ({start.date()} to {end.date()}) [{progress:.1f}%]..."
            )
            try:
                return await self._calculate_breadth_in_db(start, end)
            except Exception as e:
                self.logger.error(f"Failed to process chunk {chunk_num}: {e}")
                return None

    async def _calculate_breadth_in_db(
        self, start_date: datetime, end_date: datetime = None
    ) -> pl.DataFrame:
        """
        OPTIMIZED: Fetches raw columns and aggregates in Polars on the M1 Max.
        This avoids the heavy JOIN/GROUP BY on the slow NAS.
        """
        # 1. Fetch Features (SMA)
        query_feat = f"""
            SELECT time, asset_id, sma_20, sma_50, sma_200 
            FROM features_daily 
            WHERE time >= '{start_date.isoformat()}'
        """
        if end_date:
            query_feat += f" AND time <= '{end_date.isoformat()}'"

        # 2. Fetch Prices (Close)
        query_price = f"""
            SELECT time, asset_id, close 
            FROM market_data_daily 
            WHERE time >= '{start_date.isoformat()}'
        """
        if end_date:
            query_price += f" AND time <= '{end_date.isoformat()}'"

        # Run queries in parallel
        # Note: We use to_thread because read_database_uri is blocking
        feat_df, price_df = await asyncio.gather(
            asyncio.to_thread(pl.read_database_uri, query_feat, self.db_url),
            asyncio.to_thread(pl.read_database_uri, query_price, self.db_url),
        )

        if feat_df.is_empty() or price_df.is_empty():
            return pl.DataFrame()

        # 3. Join & Aggregate in Memory (Fast on M1)
        # Polars makes this instant
        merged = feat_df.join(price_df, on=["time", "asset_id"], how="inner")

        breadth = (
            merged.group_by("time")
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

        return breadth

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
        query = f"""
            SELECT fd.*, mdd.close
            FROM features_daily fd
            JOIN asset_metadata a ON fd.asset_id = a.id
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = a.id
            WHERE a.symbol = '{symbol}' AND fd.time >= '{start_date.isoformat()}'
            ORDER BY fd.time ASC
        """
        return await asyncio.to_thread(pl.read_database_uri, query, self.db_url)

    async def _write_context_to_db(self, df: pl.DataFrame):
        """High-Performance Upsert for MarketContextDaily with progress logging."""
        total_rows = len(df)
        self.logger.info(f"Preparing to upsert {total_rows} rows to the database...")

        cols = ["time"] + [c for c in df.columns if c != "time"]
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
            self.logger.success("Database write complete.")
