# apps/ingest-worker/engine.py

import asyncio
from collections import defaultdict
import math
import pandas as pd
import polars as pl
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, func, text, update
from sqlalchemy.dialects.postgresql import insert
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
                select(Asset.id, Asset.symbol).where(Asset.is_active == True)
            )
            self.active_assets_map = {row[0]: row[1] for row in result}

        self.logger.success(
            f"Metadata complete. {len(self.active_assets_map)} active assets loaded."
        )

    async def run_daily_ingestion(self):
        self.logger.info("--- Phase 2: Daily Data Ingestion ---")

        # 1. Safety Check: Ensure we have the asset map loaded
        # This fixes the "Temporal Coupling" brittleness
        await self._load_active_assets_from_db()

        if not self.active_assets_map:
            self.logger.warning(
                "Aborting Daily Ingestion (No active assets found in DB)."
            )
            return

        # 2. Get Latest Timestamps for all active assets
        # We only care about assets that are currently marked active
        self.logger.info("Fetching latest timestamps from database...")
        async with get_db_session() as session:
            result = await session.execute(
                select(MarketDataDaily.asset_id, func.max(MarketDataDaily.time))
                .where(MarketDataDaily.asset_id.in_(self.active_assets_map.keys()))
                .group_by(MarketDataDaily.asset_id)
            )
            latest_map = {row[0]: row[1] for row in result}

        # 3. Define Time Boundaries
        # We fetch up to yesterday to avoid SIP "15-minute delay" restrictions on Daily bars
        end_dt = get_market_close_yesterday()
        default_start = end_dt - timedelta(days=settings.ingestion.default_history_days)

        # 4. Pre-calculate Trading Schedule
        # This allows us to count actual trading days instead of calendar days
        self.logger.info("Generating trading schedule for accurate batch sizing...")
        trading_schedule = get_full_trading_schedule(
            default_start.date(), end_dt.date()
        )

        # 5. Group Assets by "Needed Start Date"
        # This optimizes requests: we don't ask for 15 years of data if a stock is only missing 1 day.
        groups = defaultdict(list)

        for asset_id, symbol in self.active_assets_map.items():
            last_ts = latest_map.get(asset_id)

            if last_ts:
                # Start from the day AFTER the last record
                needed_start = ensure_utc_timestamp(last_ts) + timedelta(days=1)
            else:
                # No data exists, start from default history (e.g. 15 years ago)
                needed_start = ensure_utc_timestamp(default_start)

            # Optimization: Ignore assets that are already up to date
            if needed_start > end_dt:
                continue

            groups[needed_start].append(asset_id)

        self.logger.info(f"Grouped assets into {len(groups)} distinct time-ranges.")

        # 6. Process Each Group
        total_groups = len(groups)
        for g_idx, (start_dt, group_asset_ids) in enumerate(groups.items(), 1):

            # A. Calculate Actual Trading Days in range
            # This is the "Staff Engineer" optimization: precise calculation
            trading_days = get_trading_days_in_range(
                start_dt.date(), end_dt.date(), trading_schedule
            )

            if trading_days <= 0:
                continue

            # B. Calculate Dynamic Batch Size
            # Formula: (Max_Datapoints * Safety_Factor) / Days_Needed
            # Example: 10,000 points / 5,000 days = 2 symbols per batch
            # Example: 10,000 points / 1 day = 10,000 symbols (clamped to max limit)
            target_datapoints = (
                settings.ingestion.max_datapoints * settings.ingestion.safety_ratio
            )
            optimum_batch_size = int(target_datapoints / trading_days)

            # Clamp: Must be at least 1, max 200 (Alpaca URL limit)
            batch_size = max(1, min(settings.ingestion.max_symbols, optimum_batch_size))

            self.logger.info(
                f"Group {g_idx}/{total_groups}: Need {trading_days} trading days "
                f"from {start_dt.date()} for {len(group_asset_ids)} assets."
            )
            self.logger.info(
                f"  -> Calculated Dynamic Batch Size: {batch_size} symbols per request."
            )

            # C. Process the Group in Batches
            for i in range(0, len(group_asset_ids), batch_size):
                batch_ids = group_asset_ids[i : i + batch_size]
                batch_symbols = [self.active_assets_map[bid] for bid in batch_ids]

                try:
                    async with self.limiter:
                        df = self.source.get_ohlcv_bars(batch_symbols, start_dt, end_dt)

                    if not df.is_empty():
                        # Use the helper to write to DB
                        count = await self._write_to_db(df, MarketDataDaily)
                        self.logger.info(
                            f"    -> Synced batch {i//batch_size + 1} | Wrote {count} rows."
                        )
                    else:
                        self.logger.debug(
                            f"    -> Batch {i//batch_size + 1} returned no data."
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error processing daily batch {batch_symbols[:3]}...: {e}"
                    )
                    # Continue to next batch, don't crash the whole worker
                    continue

    async def run_intraday_ingestion(self):
        self.logger.info("--- Phase 3: Intraday Ingestion (Trading-Day Optimized) ---")

        # 1. Safety Check: Ensure we have the asset map loaded
        await self._load_active_assets_from_db()

        if not self.active_assets_map:
            self.logger.warning("Aborting Intraday Ingestion (No active assets found).")
            return

        # 2. Resolve Configuration & Database Model
        # We don't assume 5-minute data; we ask the config what we are tracking.
        tf_unit = settings.ingestion.intraday_timeframe_unit  # e.g. "Minute"
        tf_val = settings.ingestion.intraday_timeframe_value  # e.g. 5

        try:
            # Dynamic Dispatch: Get the correct SQLAlchemy model class
            ModelClass = get_model_for_timeframe(tf_unit, tf_val)
            self.logger.info(f"Target Database Table: {ModelClass.__tablename__}")
        except ValueError as e:
            self.logger.error(f"Configuration Error: {e}")
            return

        # Map config string to Alpaca SDK Enum
        tf_enum_map = {
            "Minute": TimeFrameUnit.Minute,
            "Hour": TimeFrameUnit.Hour,
            "Day": TimeFrameUnit.Day,
        }
        timeframe = TimeFrame(tf_val, tf_enum_map[tf_unit])

        # 3. Calculate Data Density (Bars per Day)
        # This helps us estimate how many days of data fit into one API request limit (10k points).
        # Standard Session is 390 minutes (9:30 AM - 4:00 PM)
        if tf_unit == "Minute":
            bars_per_day_estimate = math.ceil(
                settings.ingestion.market_session_minutes / tf_val
            )
        elif tf_unit == "Hour":
            bars_per_day_estimate = math.ceil(
                (settings.ingestion.market_session_minutes / 60) / tf_val
            )
        else:
            bars_per_day_estimate = 1

        # 4. Get Latest Timestamps from the *Dynamic* Table
        self.logger.info(
            f"Fetching latest timestamps from {ModelClass.__tablename__}..."
        )
        async with get_db_session() as session:
            result = await session.execute(
                select(ModelClass.asset_id, func.max(ModelClass.time))
                .where(ModelClass.asset_id.in_(self.active_assets_map.keys()))
                .group_by(ModelClass.asset_id)
            )
            latest_map = {row[0]: row[1] for row in result}

        # 5. Define Global Time Boundaries
        # End date: Now minus SIP delay buffer (e.g. 15 mins) to ensure data availability
        end_dt = datetime.now(timezone.utc) - timedelta(
            minutes=settings.ingestion.sip_delay_minutes
        )
        # Start date: Lookback setting (e.g., 3 Years / 1095 days)
        default_start = end_dt - timedelta(
            days=settings.ingestion.intraday_lookback_days
        )

        # 6. Pre-calculate Trading Schedule
        # We need this to jump over weekends/holidays efficiently in the chunking loop
        self.logger.info("Generating trading schedule for accurate chunking...")
        full_schedule = get_full_trading_schedule(default_start.date(), end_dt.date())

        # 7. Batching Strategy (Symbol Batching)
        # For Intraday, we fix the Number of Symbols (to keep URL short) and vary the Time Range.
        SYMBOL_BATCH_SIZE = 5

        # We will accumulate this many API responses before writing to DB
        WRITE_BUFFER_LIMIT = 20  # 20 * 10k rows = 200k rows per write. Easy for Pi RAM.
        asset_ids = sorted(list(self.active_assets_map.keys()))

        for i in range(0, len(asset_ids), SYMBOL_BATCH_SIZE):
            batch_ids = asset_ids[i : i + SYMBOL_BATCH_SIZE]
            batch_symbols = [self.active_assets_map[bid] for bid in batch_ids]

            # A. Determine Earliest Needed Start for this Batch
            batch_needed_start = end_dt
            for bid in batch_ids:
                last = latest_map.get(bid)
                if last:
                    # Start 1 interval after the last bar to avoid duplicates
                    delta = (
                        timedelta(minutes=tf_val)
                        if tf_unit == "Minute"
                        else timedelta(hours=tf_val)
                    )
                    needed = ensure_utc_timestamp(last + delta)
                else:
                    needed = ensure_utc_timestamp(default_start)

                if needed < batch_needed_start:
                    batch_needed_start = needed

            if batch_needed_start >= end_dt:
                continue

            # B. Calculate Max Trading Days per Request (Dynamic Chunk Size)
            # Formula: (Max_Limit * Safety) / (Num_Symbols * Bars_Per_Day)
            target_datapoints = (
                settings.ingestion.max_datapoints * settings.ingestion.safety_ratio
            )
            daily_load_per_batch = len(batch_symbols) * bars_per_day_estimate

            if daily_load_per_batch == 0:
                daily_load_per_batch = 1

            max_trading_days_per_req = math.floor(
                target_datapoints / daily_load_per_batch
            )
            # Clamp: At least 1 day, max 30 days (to avoid crazy long timeouts/processing)
            chunk_trading_days = max(1, min(30, max_trading_days_per_req))

            self.logger.info(
                f"Batch {i//SYMBOL_BATCH_SIZE + 1}: {len(batch_symbols)} syms. "
                f"Range: {batch_needed_start.date()} -> {end_dt.date()}. "
                f"Chunk Size: {chunk_trading_days} trading days."
            )

            # C. Time Chunk Loop (Using Schedule)
            # Filter schedule for the relevant range for this specific batch
            # We must use proper timestamp comparison
            relevant_schedule = full_schedule[
                (full_schedule.index >= pd.Timestamp(batch_needed_start.date()))
                & (full_schedule.index <= pd.Timestamp(end_dt.date()))
            ]

            if relevant_schedule.empty:
                continue

            schedule_dates = relevant_schedule.index.to_list()

            # --- BUFFERING LOGIC START ---
            data_buffer = []

            # Iterate through the schedule in chunks of 'chunk_trading_days'
            for j in range(0, len(schedule_dates), chunk_trading_days):
                chunk_dates = schedule_dates[j : j + chunk_trading_days]

                # 1. Get Naive/Raw dates from schedule
                raw_chunk_start = chunk_dates[0]
                raw_chunk_end = chunk_dates[-1] + timedelta(days=1)

                # 2. Convert to UTC-Aware Datetimes
                chunk_start_utc = ensure_utc_timestamp(raw_chunk_start)
                chunk_end_utc = ensure_utc_timestamp(raw_chunk_end)

                # 3. Calculate Request Boundaries
                # Now both variables in min() are UTC-aware, so no error.
                req_start = batch_needed_start if j == 0 else chunk_start_utc
                req_end = min(chunk_end_utc, end_dt)

                if req_start >= req_end:
                    continue

                self.logger.debug(
                    f"  -> Fetching Chunk: {req_start.date()} to {req_end.date()}"
                )

                try:
                    async with self.limiter:
                        df = self.source.get_ohlcv_bars(
                            symbols=batch_symbols,
                            start_dt=req_start,
                            end_dt=req_end,
                            timeframe=timeframe,
                        )

                    if not df.is_empty():
                        data_buffer.append(df)

                    # Check Buffer Size
                    if len(data_buffer) >= WRITE_BUFFER_LIMIT:
                        await self._flush_buffer(data_buffer, ModelClass)
                        data_buffer = []  # Clear buffer

                    # if not df.is_empty():
                    #     count = await self._write_to_db(df, ModelClass)
                    #     self.logger.info(f"     -> Wrote {count} rows.")
                except Exception as e:
                    self.logger.error(f"Error fetching intraday chunk: {e}")

            # End of Batch: Flush remaining data
            if data_buffer:
                await self._flush_buffer(data_buffer, ModelClass)

    async def _write_to_db(self, df: pl.DataFrame, model_class) -> int:
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

        async with get_db_session() as session:
            # --- THE FIX STARTS HERE ---
            # 1. Get the SQLAlchemy AsyncConnection
            conn = await session.connection()

            # 2. Await the method to get the DBAPI proxy
            dbapi_conn = await conn.get_raw_connection()

            # 3. Access the actual asyncpg driver connection
            asyncpg_conn = dbapi_conn.driver_connection
            # --- THE FIX ENDS HERE ---

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
            await session.commit()

        return total_rows

    async def _flush_buffer(self, buffer_list: list[pl.DataFrame], model_class):
        if not buffer_list:
            return

        self.logger.info(
            f"  >> Flushing Buffer: Writing {len(buffer_list)} chunks to DB..."
        )

        # 1. Concatenate all dataframes in memory (Fast on Pi)
        combined_df = pl.concat(buffer_list)

        # 2. Write once
        count = await self._write_to_db(combined_df, model_class)
        self.logger.success(f"     -> Wrote {count} rows in one transaction.")

    async def _load_active_assets_from_db(self):
        """
        Internal helper: Loads the active asset map from the DB into memory
        if it hasn't been loaded yet.
        """
        if self.active_assets_map:
            return  # Already loaded, do nothing

        self.logger.info("Asset map empty. Loading active assets from Database...")
        async with get_db_session() as session:
            result = await session.execute(
                select(Asset.id, Asset.symbol).where(Asset.is_active == True)
            )
            self.active_assets_map = {row[0]: row[1] for row in result}

        if not self.active_assets_map:
            self.logger.warning(
                "Database contains NO active assets. Did you run metadata sync?"
            )
        else:
            self.logger.info(f"Loaded {len(self.active_assets_map)} assets from DB.")
