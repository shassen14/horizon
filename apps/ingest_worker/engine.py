# apps/ingest_worker/engine.py

import asyncio
from collections import defaultdict
import math
import pandas as pd
import polars as pl
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, func, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm.attributes import InstrumentedAttribute
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from aiolimiter import AsyncLimiter

# We import types for type hinting only
from packages.quant_lib.config import settings
from packages.database.session import get_db_session
from packages.database.models import (
    Asset,
    MarketDataDaily,
    get_model_for_timeframe,
)
from packages.quant_lib.date_utils import (
    ensure_utc_timestamp,
    get_market_close_yesterday,
    get_full_trading_schedule,
    get_trading_days_in_range,
)

from packages.quant_lib.interfaces import DataSource


class IngestionEngine:
    def __init__(self, source: DataSource, logger, limiter: AsyncLimiter):
        self.source = source
        self.logger = logger
        self.limiter = limiter

        # Internal state
        self.active_assets_map: dict[int, str] = {}

        # Concurrency Control
        # 5 concurrent batches is the sweet spot for the Pi 5.
        # It balances RAM usage (DataFrame buffers) vs CPU/IO throughput.
        self.concurrency_limiter = asyncio.Semaphore(5)

    async def run_metadata_sync(self):
        """
        Smart Metadata Sync:
        1. Checks if DB metadata is fresh.
        2. If fresh -> Loads state from DB into self.active_assets_map.
        3. If stale -> Runs Probe -> Updates DB -> Loads state into self.active_assets_map.
        """
        self.logger.info("--- Phase 1: Metadata Sync & Screener ---")

        # --- A. STALENESS CHECK ---
        async with get_db_session() as session:
            result = await session.execute(func.max(Asset.last_updated))
            last_update = result.scalar()

        should_run_screener = True
        if last_update:
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - last_update

            if age < timedelta(hours=settings.ingestion.metadata_cache_hours):
                self.logger.info(
                    f"Metadata is fresh ({age.total_seconds()//3600:.1f}h old)."
                )
                self.logger.warning("SKIPPING heavy screener/probe.")
                should_run_screener = False
            else:
                self.logger.info("Metadata is stale. Running full update.")
        else:
            self.logger.info("No metadata found (First run). Running full update.")

        # --- B. FAST PATH (Skip Probe) ---
        if not should_run_screener:
            async with get_db_session() as session:
                # Load existing active assets directly into state
                result = await session.execute(
                    select(Asset.id, Asset.symbol).where(Asset.is_active == True)
                )
                self.active_assets_map = {row[0]: row[1] for row in result}

            self.logger.success(
                f"Loaded {len(self.active_assets_map)} existing active assets from DB."
            )
            return  # Exit here, no need to return a value

        # --- C. SLOW PATH (Full Probe) ---
        # 1. Fetch Metadata
        all_assets = self.source.get_all_tickers()
        self.logger.info(f"Alpaca returned {len(all_assets)} total tradable assets.")

        # 2. Filter (Memory)
        allowed_exchanges = settings.screener.allowed_exchanges_list
        blocked_exchanges = settings.screener.blocked_exchanges_list
        target_classes = settings.screener.asset_classes_list
        BLOCKED_PATTERNS = [".", "$"]

        candidates = []
        for a in all_assets:
            if a.get("asset_class") not in target_classes:
                continue
            if allowed_exchanges and a["exchange"] not in allowed_exchanges:
                continue
            if blocked_exchanges and a["exchange"] in blocked_exchanges:
                continue
            if any(p in a["symbol"] for p in BLOCKED_PATTERNS):
                continue
            candidates.append(a)

        # 3. Data Filter (The Probe)
        self.logger.info(f"Starting Data Probe on {len(candidates)} candidates...")
        end_dt = get_market_close_yesterday()
        start_dt = end_dt - timedelta(days=5)

        PROBE_BATCH_SIZE = 200
        candidate_symbols = [a["symbol"] for a in candidates]
        valid_symbols_set = set()

        for i in range(0, len(candidate_symbols), PROBE_BATCH_SIZE):
            batch = candidate_symbols[i : i + PROBE_BATCH_SIZE]
            df = self.source.get_ohlcv_bars(batch, start_dt, end_dt)

            if df.is_empty():
                continue

            try:
                stats = (
                    df.lazy()
                    .with_columns(
                        [(pl.col("close") * pl.col("volume")).alias("dollar_vol")]
                    )
                    .group_by("symbol")
                    .agg(
                        [
                            pl.col("dollar_vol").mean().alias("avg_dollar_vol"),
                            pl.col("close").last().alias("last_price"),
                        ]
                    )
                    .filter(
                        (pl.col("last_price") >= settings.screener.min_price)
                        & (pl.col("avg_dollar_vol") >= settings.screener.min_dollar_vol)
                    )
                    .collect()
                )
                valid_symbols_set.update(stats["symbol"].to_list())
            except Exception as e:
                self.logger.error(f"Error filtering batch {i}: {e}")

            self.logger.info(
                f"  Probe Batch {i//PROBE_BATCH_SIZE + 1}: {len(batch)} -> {len(stats)} qualified."
            )

        # --- D. UPDATE DATABASE & STATE ---
        self.logger.info(
            f"Updating Database with {len(valid_symbols_set)} active assets..."
        )

        quality_assets = [a for a in candidates if a["symbol"] in valid_symbols_set]
        active_symbols_list = [a["symbol"] for a in quality_assets]

        async with get_db_session() as session:
            # 1. Reset ALL to inactive
            await session.execute(update(Asset).values(is_active=False))

            # 2. Get IDs of existing assets
            result = await session.execute(
                select(Asset.symbol, Asset.id).where(
                    Asset.symbol.in_(active_symbols_list)
                )
            )
            existing_map = {row[0]: row[1] for row in result}

            # 3. Mark Existing as Active
            existing_ids = list(existing_map.values())
            if existing_ids:
                # Chunk the update just in case
                for i in range(0, len(existing_ids), 5000):
                    chunk = existing_ids[i : i + 5000]
                    await session.execute(
                        update(Asset).where(Asset.id.in_(chunk)).values(is_active=True)
                    )

            # 4. Insert New Assets
            new_assets = [a for a in quality_assets if a["symbol"] not in existing_map]
            if new_assets:
                records = [
                    {
                        "symbol": a["symbol"],
                        "name": a["name"],
                        "exchange": a["exchange"],
                        "asset_class": a["asset_class"],
                        "is_active": True,
                        "last_updated": datetime.now(timezone.utc),
                    }
                    for a in new_assets
                ]
                INSERT_CHUNK_SIZE = 5000
                for i in range(0, len(records), INSERT_CHUNK_SIZE):
                    chunk = records[i : i + INSERT_CHUNK_SIZE]
                    self.logger.info(
                        f"  Writing Asset Metadata Chunk {i//INSERT_CHUNK_SIZE + 1}..."
                    )
                    await session.execute(insert(Asset), chunk)
                    await session.commit()  # Commit per chunk to free memory/locks

            await session.commit()

            # 5. FINAL STEP: Populate Class State
            # We re-query to get the authoritative list of IDs and Symbols
            result = await session.execute(
                select(Asset.id, Asset.symbol)
                .where(Asset.is_active == True)
                .order_by(Asset.symbol)
            )
            self.active_assets_map = {row[0]: row[1] for row in result}

        self.logger.success(
            f"Metadata complete. {len(self.active_assets_map)} active assets loaded."
        )

    async def run_daily_ingestion(self):
        self.logger.info("--- Phase 2: Daily Data Ingestion (Concurrent) ---")

        # Lazy load assets
        await self._ensure_assets_loaded()

        if not self.active_assets_map:
            self.logger.warning("No active assets found. Skipping Daily Ingestion.")
            return

        # 1. Get Latest Timestamps
        async with get_db_session() as session:
            # Reading ledger for last update
            result = await session.execute(
                select(Asset.id, Asset.last_market_data_daily_update).where(
                    Asset.id.in_(self.active_assets_map.keys())
                )
            )
            latest_map = {row[0]: row[1] for row in result}

        end_dt = get_market_close_yesterday()
        default_start = end_dt - timedelta(days=settings.ingestion.default_history_days)

        trading_schedule = get_full_trading_schedule(
            default_start.date(), end_dt.date()
        )

        # 2. Group Assets
        groups = defaultdict(list)
        for asset_id, symbol in self.active_assets_map.items():
            last_ts = latest_map.get(asset_id)
            needed = (
                ensure_utc_timestamp(last_ts + timedelta(days=1))
                if last_ts
                else ensure_utc_timestamp(default_start)
            )

            if needed > end_dt:
                continue
            groups[needed].append(asset_id)

        # 3. Create Tasks
        tasks = []

        for start_dt, group_asset_ids in groups.items():
            # Calculate Dynamic Batch Size
            trading_days = get_trading_days_in_range(
                start_dt.date(), end_dt.date(), trading_schedule
            )
            if trading_days <= 0:
                continue

            target_dp = (
                settings.ingestion.max_datapoints * settings.ingestion.safety_ratio
            )
            optimum_batch_size = int(target_dp / trading_days)
            batch_size = max(1, min(settings.ingestion.max_symbols, optimum_batch_size))

            # Sorted ensures determinism
            sorted_group_asset_ids = sorted(
                group_asset_ids, key=lambda aid: self.active_assets_map[aid]
            )

            for i in range(0, len(sorted_group_asset_ids), batch_size):
                batch_ids = group_asset_ids[i : i + batch_size]

                # Spawn Task
                task = asyncio.create_task(
                    self._process_daily_batch(batch_ids, start_dt, end_dt)
                )
                tasks.append(task)

        if tasks:
            self.logger.info(f"Queued {len(tasks)} daily batches. Processing...")
            await asyncio.gather(*tasks)

    async def _process_daily_batch(self, batch_ids, start_dt, end_dt):
        """Worker function for a single Daily batch."""
        async with self.concurrency_limiter:
            # Create a short, readable representation of the batch
            batch_symbols = [self.active_assets_map[bid] for bid in batch_ids]

            # Show the first 2 symbols and the total count for brevity
            log_context = f"Batch ({batch_symbols[0]}... {len(batch_symbols)} total)"

            # Log the "What" and "Why"
            self.logger.info(
                f"{log_context}: Fetching from {start_dt.date()} to {end_dt.date()}"
            )

            import time

            fetch_start_time = time.time()

            # Rate Limit Logic
            async with self.limiter:
                df = self.source.get_ohlcv_bars(batch_symbols, start_dt, end_dt)

            fetch_duration = time.time() - fetch_start_time

            if df.is_empty():
                self.logger.warning(f"{log_context}: No data returned from source.")
                return

            write_start_time = time.time()

            count = await self._write_to_db(
                df, MarketDataDaily, Asset.last_market_data_daily_update
            )

            write_duration = time.time() - write_start_time

            self.logger.success(
                f"{log_context}: Success. "
                f"Wrote {count} rows. "
                f"Fetch: {fetch_duration:.2f}s, Write: {write_duration:.2f}s"
            )

    async def run_intraday_ingestion(self):
        self.logger.info("--- Phase 3: Intraday Ingestion (Concurrent) ---")

        # Lazy load assets
        await self._ensure_assets_loaded()

        if not self.active_assets_map:
            self.logger.warning("No active assets found. Skipping Intraday Ingestion.")
            return

        # 1. Config & Model
        tf_unit = settings.ingestion.intraday_timeframe_unit
        tf_val = settings.ingestion.intraday_timeframe_value
        try:
            ModelClass = get_model_for_timeframe(tf_unit, tf_val)
        except ValueError:
            return

        # Map to Alpaca Enum
        tf_enum_map = {"Minute": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour}
        timeframe = TimeFrame(tf_val, tf_enum_map[tf_unit])

        # 2. Latest Map
        async with get_db_session() as session:
            # Reading Ledger
            result = await session.execute(
                select(Asset.id, Asset.last_market_data_5min_update).where(
                    Asset.id.in_(self.active_assets_map.keys())
                )
            )
            latest_map = {row[0]: row[1] for row in result}

        end_dt = datetime.now(timezone.utc) - timedelta(
            minutes=settings.ingestion.sip_delay_minutes
        )
        default_start = end_dt - timedelta(
            days=settings.ingestion.intraday_lookback_days
        )

        trading_schedule = get_full_trading_schedule(
            default_start.date(), end_dt.date()
        )

        # 3. Spawn Tasks
        tasks = []
        SYMBOL_BATCH_SIZE = 50
        # We sort the entire list of IDs based on their symbol alphabetically before creating batches.
        asset_ids = sorted(
            list(self.active_assets_map.keys()),
            key=lambda aid: self.active_assets_map[aid],
        )

        for i in range(0, len(asset_ids), SYMBOL_BATCH_SIZE):
            batch_ids = asset_ids[i : i + SYMBOL_BATCH_SIZE]

            task = asyncio.create_task(
                self._process_intraday_batch(
                    batch_ids,
                    latest_map,
                    full_schedule=trading_schedule,
                    default_start=default_start,
                    end_dt=end_dt,
                    timeframe=timeframe,
                    ModelClass=ModelClass,
                )
            )
            tasks.append(task)

        if tasks:
            self.logger.info(f"Queued {len(tasks)} intraday batches. Processing...")
            await asyncio.gather(*tasks)

    async def _process_intraday_batch(
        self,
        batch_ids,
        latest_map,
        full_schedule,
        default_start,
        end_dt,
        timeframe,
        ModelClass,
    ):
        """Worker function for a single Intraday batch with Time Chunking."""
        async with self.concurrency_limiter:
            batch_symbols = [self.active_assets_map[bid] for bid in batch_ids]

            # Determine start date
            batch_needed_start = end_dt
            for bid in batch_ids:
                last = latest_map.get(bid)
                needed = (
                    ensure_utc_timestamp(
                        last
                        + timedelta(minutes=settings.ingestion.intraday_timeframe_value)
                    )
                    if last
                    else default_start
                )
                if needed < batch_needed_start:
                    batch_needed_start = needed

            if batch_needed_start >= end_dt:
                return

            # Calculate Chunk Size
            target_dp = (
                settings.ingestion.max_datapoints * settings.ingestion.safety_ratio
            )
            # Estimate bars per day (e.g. 78 for 5min)
            bars_per_day = math.ceil(
                settings.ingestion.market_session_minutes
                / settings.ingestion.intraday_timeframe_value
            )

            daily_load = len(batch_symbols) * bars_per_day
            max_days = math.floor(target_dp / daily_load)
            chunk_days_limit = max(1, min(30, max_days))

            # Chunk Loop
            relevant_schedule = full_schedule[
                (full_schedule.index >= pd.Timestamp(batch_needed_start.date()))
                & (full_schedule.index <= pd.Timestamp(end_dt.date()))
            ]

            if relevant_schedule.empty:
                return

            schedule_dates = relevant_schedule.index.to_list()

            data_buffer = []  # In-Memory Buffer for efficient writes

            for j in range(0, len(schedule_dates), chunk_days_limit):
                chunk_dates = schedule_dates[j : j + chunk_days_limit]

                # Get naive/pandas dates
                raw_chunk_start = chunk_dates[0]
                raw_chunk_end = chunk_dates[-1] + timedelta(days=1)

                chunk_start_utc = ensure_utc_timestamp(raw_chunk_start)
                chunk_end_utc = ensure_utc_timestamp(raw_chunk_end)

                req_start = batch_needed_start if j == 0 else chunk_start_utc
                req_end = min(chunk_end_utc, end_dt)

                if req_start >= req_end:
                    continue

                self.logger.info(
                    f"Fetching {batch_symbols[0]}... ({len(batch_symbols)} syms) | {req_start.date()} -> {req_end.date()}"
                )

                async with self.limiter:
                    df = self.source.get_ohlcv_bars(
                        batch_symbols, req_start, req_end, timeframe
                    )

                if not df.is_empty():
                    data_buffer.append(df)

                # Flush Buffer if full (e.g. 20 chunks = ~200k rows)
                if len(data_buffer) >= 20:
                    await self._flush_buffer(
                        data_buffer, ModelClass, Asset.last_market_data_5min_update
                    )
                    data_buffer = []

            # Final Flush
            if data_buffer:
                await self._flush_buffer(
                    data_buffer, ModelClass, Asset.last_market_data_5min_update
                )

    async def _write_to_db(
        self, df: pl.DataFrame, model_class, ledger_col: InstrumentedAttribute
    ) -> int:
        """
        High-Performance Write using PostgreSQL COPY protocol via a Temp Table.
        """
        if df.is_empty():
            return 0

        # 1. Map Symbol string back to Asset ID integer
        asset_ids = list(self.active_assets_map.keys())
        symbols = list(self.active_assets_map.values())
        map_df = pl.DataFrame({"symbol": symbols, "asset_id": asset_ids})

        df = df.join(map_df, on="symbol", how="left")

        # 2. Select columns
        target_cols = [
            "time",
            "asset_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
        ]
        valid_cols = [c for c in target_cols if c in df.columns]

        df = df.select(valid_cols)

        # Convert to List of Tuples (Required for asyncpg COPY)
        records = list(df.iter_rows())

        if not records:
            return 0

        total_rows = len(records)
        table_name = model_class.__tablename__

        # Calculate Max Times for Ledger
        max_times = (
            df.group_by("asset_id")
            .agg(pl.col("time").max().alias("new_max"))
            .select(["asset_id", "new_max"])
            .to_dicts()
        )

        async with get_db_session() as session:
            # Speed Hack: No sync commit
            await session.execute(text("SET LOCAL synchronous_commit TO OFF;"))

            # 1. Get the SQLAlchemy AsyncConnection
            conn = await session.connection()

            # 2. Await the method to get the DBAPI proxy
            dbapi_conn = await conn.get_raw_connection()

            # 3. Access the actual asyncpg driver connection
            asyncpg_conn = dbapi_conn.driver_connection

            # A. Create a Temporary Staging Table
            temp_table = f"temp_{table_name}_ingest"

            await session.execute(
                text(
                    f"""
                CREATE TEMP TABLE IF NOT EXISTS {temp_table} 
                (LIKE "{table_name}" INCLUDING DEFAULTS)
                ON COMMIT DROP;
            """
                )
            )

            # B. COPY data into the Temp Table (using the raw driver)
            try:
                await asyncpg_conn.copy_records_to_table(
                    temp_table, records=records, columns=valid_cols
                )
            except Exception as e:
                self.logger.error(f"COPY failed: {e}")
                raise e

            # C. Move from Temp to Real Table
            columns_str = ", ".join([f'"{c}"' for c in valid_cols])

            insert_query = f"""
                INSERT INTO "{table_name}" ({columns_str})
                SELECT {columns_str} FROM {temp_table}
                ON CONFLICT (time, asset_id) DO NOTHING;
            """

            await session.execute(text(insert_query))

            for row in max_times:
                aid = row["asset_id"]
                new_ts = row["new_max"]

                # Ensure timezone awareness
                if new_ts.tzinfo is None:
                    new_ts = new_ts.replace(tzinfo=timezone.utc)

                # Use the column OBJECT as the key in the dictionary
                stmt = update(Asset).where(Asset.id == aid).values({ledger_col: new_ts})
                await session.execute(stmt)

            # Commit both the Data Insert and the Ledger Update together
            await session.commit()

        return total_rows

    async def _flush_buffer(
        self,
        buffer_list: list[pl.DataFrame],
        model_class,
        ledger_col: InstrumentedAttribute,
    ):
        if not buffer_list:
            return

        self.logger.info(
            f"  >> Flushing Buffer: Writing {len(buffer_list)} chunks to DB..."
        )

        # 1. Concatenate all dataframes in memory (Fast on Pi)
        combined_df = pl.concat(buffer_list)

        # 2. Write once
        count = await self._write_to_db(combined_df, model_class, ledger_col)
        self.logger.success(f"     -> Wrote {count} rows in one transaction.")

    async def _load_active_assets(self, specific_symbol: str | None = None):
        """
        Internal helper: Loads the map of assets marked as active from the database.
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
                f"Loaded {len(self.active_assets_map)} active assets to process."
            )

    async def _ensure_assets_loaded(self):
        """
        Lazy Loader: Checks if assets are loaded and loads them if not.
        This is the single entry point for checking asset state.
        """
        if not self.active_assets_map:
            await self._load_active_assets()
