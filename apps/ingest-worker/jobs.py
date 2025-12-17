# apps/ingest-worker/jobs.py

import asyncio
from collections import defaultdict
import math
import polars as pl
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, func, update
from sqlalchemy.dialects.postgresql import insert
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from packages.quant_lib.config import settings
from packages.quant_lib.date_utils import (
    ensure_utc_timestamp,
    get_full_trading_schedule,
    get_market_close_yesterday,
    get_trading_days_in_range,
)
from packages.database.session import get_db_session
from packages.database.models import (
    Asset,
    MarketDataDaily,
    get_model_for_timeframe,
)
from sources.alpaca import AlpacaSource


# --- Phase 1: Metadata & Screener ---
async def run_metadata_sync(source: AlpacaSource) -> dict[int, str]:
    """
    Smart Metadata Sync:
    1. Checks if DB metadata is fresh (updated < 24h ago).
    2. If fresh -> Returns existing ACTIVE assets immediately (Skips Probe).
    3. If stale -> Runs full Probe (Fetch All -> Filter -> Probe Data -> Update DB).
    """
    print("--- Phase 1: Metadata Sync & Screener ---")

    # --- A. STALENESS CHECK ---
    async with get_db_session() as session:
        # Get the most recent update time from the DB
        result = await session.execute(func.max(Asset.last_updated))
        last_update = result.scalar()

    # Determine if we should skip
    should_run_screener = True
    if last_update:
        # Calculate how old the data is
        # Ensure last_update is timezone-aware (it should be stored as UTC)
        if last_update.tzinfo is None:
            last_update = last_update.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age = now - last_update

        if age < timedelta(hours=settings.METADATA_CACHE_HOURS):
            print(
                f"Metadata is fresh (Last updated {age.seconds//3600}h {age.seconds%3600//60}m ago)."
            )
            print("SKIPPING heavy screener/probe.")
            should_run_screener = False
        else:
            print(
                f"Metadata is stale (Last updated {age.days} days ago). Running full update."
            )
    else:
        print("No metadata found (First run). Running full update.")

    # --- B. FAST PATH (Skip Probe) ---
    if not should_run_screener:
        async with get_db_session() as session:
            # Just return what we already have marked as active
            result = await session.execute(
                select(Asset.id, Asset.symbol).where(Asset.is_active == True)
            )
            active_map = {row[0]: row[1] for row in result}
            print(f"Loaded {len(active_map)} existing active assets from DB.")
            return active_map

    # 1. Fetch Metadata (All Tradable Assets)
    all_assets = source.get_all_tickers()
    print(f"Alpaca returned {len(all_assets)} total tradable assets.")

    # 2. Metadata Filter (Using SETTINGS)
    # We use the computed list properties from our settings
    allowed_exchanges = settings.ASSET_SCREENER_EXCHANGES_ALLOWED_LIST
    blocked_exchanges = settings.ASSET_SCREENER_EXCHANGES_BLOCKED_LIST
    target_asset_classes = settings.ASSET_SCREENER_CLASSES_LIST

    # Hardcoded heuristics for "bad" symbols (warrants/preferreds)
    # You could add this to config, but it's usually static logic
    BLOCKED_PATTERNS = [".", "$"]

    candidates = []
    for a in all_assets:
        # Filter 1: Asset Class (e.g., 'us_equity')
        if a.get("asset_class") not in target_asset_classes:
            continue

        # Filter 2: Exchanges
        if allowed_exchanges and a["exchange"] not in allowed_exchanges:
            continue
        if blocked_exchanges and a["exchange"] in blocked_exchanges:
            continue

        # Filter 3: Symbol Patterns (remove warrants like 'ABC.W')
        if any(p in a["symbol"] for p in BLOCKED_PATTERNS):
            continue

        candidates.append(a)

    print(f"Metadata filtering reduced list to {len(candidates)} candidates.")

    # 3. Data Filter (The "Probe")
    print("Starting Data Probe (checking Price & Volume)...")
    end_dt = get_market_close_yesterday()
    start_dt = end_dt - timedelta(days=5)

    PROBE_BATCH_SIZE = 200
    candidate_symbols = [a["symbol"] for a in candidates]
    valid_symbols_set = set()

    for i in range(0, len(candidate_symbols), PROBE_BATCH_SIZE):
        batch = candidate_symbols[i : i + PROBE_BATCH_SIZE]

        # Fetch 5 days of bars
        df = source.get_ohlcv_bars(batch, start_dt, end_dt)

        if df.is_empty():
            continue

        # POLARS FILTERING
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
                    (pl.col("last_price") >= settings.SCREENER_MIN_PRICE)
                    & (pl.col("avg_dollar_vol") >= settings.SCREENER_MIN_DOLLAR_VOLUME)
                )
                .collect()
            )
            valid_symbols_set.update(stats["symbol"].to_list())
        except Exception as e:
            print(f"Error filtering batch {i}: {e}")

        print(
            f"  Probe Batch {i//PROBE_BATCH_SIZE + 1}: {len(batch)} -> {len(stats)} qualified."
        )

    print(f"Final Quality List: {len(valid_symbols_set)} assets.")

    # 4. Update Database
    quality_assets = [a for a in candidates if a["symbol"] in valid_symbols_set]
    active_symbols = [a["symbol"] for a in quality_assets]

    async with get_db_session() as session:
        # A. Reset ALL to inactive
        await session.execute(update(Asset).values(is_active=False))

        # B. Get existing IDs
        result = await session.execute(
            select(Asset.symbol, Asset.id).where(Asset.symbol.in_(active_symbols))
        )
        existing_map = {row[0]: row[1] for row in result}

        # C. Update existing to ACTIVE
        # Chunking here is usually not needed for updates unless massive,
        # but the WHERE IN clause has a limit too (usually 65k parameters).
        # Safe to chunk updates if list is > 10k.
        existing_ids = list(existing_map.values())
        UPDATE_CHUNK_SIZE = 5000
        if existing_ids:
            for i in range(0, len(existing_ids), UPDATE_CHUNK_SIZE):
                chunk_ids = existing_ids[i : i + UPDATE_CHUNK_SIZE]
                await session.execute(
                    update(Asset).where(Asset.id.in_(chunk_ids)).values(is_active=True)
                )

        # D. Insert NEW assets (CRITICAL FIX: CHUNKING ADDED)
        new_assets = [a for a in quality_assets if a["symbol"] not in existing_map]

        if new_assets:
            print(f"Inserting {len(new_assets)} new assets into DB...")
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

            # --- CHUNKING LOGIC ---
            INSERT_CHUNK_SIZE = 5000
            for i in range(0, len(records), INSERT_CHUNK_SIZE):
                chunk = records[i : i + INSERT_CHUNK_SIZE]
                print(f"  Writing Asset Metadata Chunk {i//INSERT_CHUNK_SIZE + 1}...")
                await session.execute(insert(Asset), chunk)
                await session.commit()  # Commit per chunk to free memory/locks

        await session.commit()

        # Return map
        result = await session.execute(
            select(Asset.id, Asset.symbol).where(Asset.is_active == True)
        )
        return {row[0]: row[1] for row in result}


# --- Phase 2: Daily Data (Horizon) ---
async def run_daily_ingestion(source: AlpacaSource, active_assets: dict[int, str]):
    print("--- Phase 2: Daily Data Ingestion (Smart Batching) ---")

    # 1. Get Latest Timestamps for ALL active assets
    async with get_db_session() as session:
        result = await session.execute(
            select(MarketDataDaily.asset_id, func.max(MarketDataDaily.time))
            .where(MarketDataDaily.asset_id.in_(active_assets.keys()))
            .group_by(MarketDataDaily.asset_id)
        )
        latest_map = {row[0]: row[1] for row in result}

    end_dt = get_market_close_yesterday()
    default_start = end_dt - timedelta(days=settings.DEFAULT_HISTORY_DAYS)

    print("Generating trading schedule for accurate day counting...")
    trading_schedule = get_full_trading_schedule(default_start.date(), end_dt.date())

    # 2. Group Assets by "Needed Start Date"
    # This prevents requesting 15 years of data for a batch that mostly only needs 1 day.
    groups = defaultdict(list)
    for asset_id, symbol in active_assets.items():
        last_ts = latest_map.get(asset_id)
        needed_start = (
            ensure_utc_timestamp(last_ts + timedelta(days=1))
            if last_ts
            else ensure_utc_timestamp(default_start)
        )

        if needed_start > end_dt:
            continue
        groups[needed_start].append(asset_id)

    print(f"Grouped assets into {len(groups)} distinct time-ranges.")

    # 3. Process each group
    for start_dt, group_asset_ids in groups.items():
        # Calculate Actual Trading Days in range
        trading_days = get_trading_days_in_range(
            start_dt.date(), end_dt.date(), trading_schedule
        )

        if trading_days <= 0:
            continue

        # Calculate Dynamic Batch Size
        # Formula: (Limit * Safety) / TradingDays
        target_datapoints = settings.ALPACA_MAX_DATAPOINTS * settings.API_SAFETY_RATIO
        optimum_batch_size = int(target_datapoints / trading_days)
        # Clamp between 1 and the Hard API Limit (200)
        batch_size = max(1, min(settings.ALPACA_MAX_SYMBOLS, optimum_batch_size))

        print(
            f"Time Range: {start_dt.date()} -> {end_dt.date()} ({trading_days} trading days)"
        )
        print(
            f"  -> Batch Size: {batch_size} symbols (for {len(group_asset_ids)} total in group)"
        )

        # 4. Iterate through the group in calculated batches
        for i in range(0, len(group_asset_ids), batch_size):
            batch_ids = group_asset_ids[i : i + batch_size]
            batch_symbols = [active_assets[bid] for bid in batch_ids]

            # Fetch
            df = source.get_ohlcv_bars(batch_symbols, start_dt, end_dt)

            if not df.is_empty():
                count = await _write_to_db(df, active_assets, MarketDataDaily)
                print(f"    -> Wrote {count} rows.")


# --- Phase 3: Intraday Data (Vector) ---
async def run_intraday_ingestion(source: AlpacaSource, active_assets: dict[int, str]):
    print("--- Phase 3: Intraday Ingestion (Dynamic & Optimized) ---")

    # 1. Resolve Configuration & Database Model
    tf_unit = settings.INTRADAY_BAR_TIMEFRAME
    tf_val = settings.INTRADAY_BAR_TIMEFRAME_VALUE

    try:
        # Dynamic Dispatch: Get the correct table (e.g., MarketData5Min)
        ModelClass = get_model_for_timeframe(tf_unit, tf_val)
        print(f"Target Table: {ModelClass.__tablename__}")
    except ValueError as e:
        print(f"CRITICAL: {e}")
        return

    # Map string config to Alpaca SDK Enum
    tf_enum_map = {
        "Minute": TimeFrameUnit.Minute,
        "Hour": TimeFrameUnit.Hour,
        "Day": TimeFrameUnit.Day,
    }
    timeframe = TimeFrame(tf_val, tf_enum_map[tf_unit])

    # 2. Calculate Data Density (Bars per Day)
    # We use settings.MARKET_SESSION_MINUTES (390) instead of magic number
    if tf_unit == "Minute":
        bars_per_day_estimate = math.ceil(settings.MARKET_SESSION_MINUTES / tf_val)
    elif tf_unit == "Hour":
        bars_per_day_estimate = math.ceil(
            (settings.MARKET_SESSION_MINUTES / 60) / tf_val
        )
    else:
        bars_per_day_estimate = 1

    print(f"Estimated data density: {bars_per_day_estimate} bars/day per symbol.")

    # 3. Get Latest Timestamps from the *Dynamic* Table
    async with get_db_session() as session:
        result = await session.execute(
            select(ModelClass.asset_id, func.max(ModelClass.time))
            .where(ModelClass.asset_id.in_(active_assets.keys()))
            .group_by(ModelClass.asset_id)
        )
        latest_map = {row[0]: row[1] for row in result}

    # 4. Define Global Time Boundaries
    # End date: Now - SIP Delay (e.g. 15 mins)
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=settings.SIP_DELAY_MINUTES)
    default_start = end_dt - timedelta(days=settings.INTRADAY_LOOKBACK_DAYS)

    # 5. Batching Strategy
    # We fix the number of symbols per request to keep URLs clean
    SYMBOL_BATCH_SIZE = 50
    asset_ids = list(active_assets.keys())

    # Pre-calculate trading schedule for the whole lookback period
    # This lets us skip weekends/holidays logic easily
    full_schedule = get_full_trading_schedule(default_start.date(), end_dt.date())

    for i in range(0, len(asset_ids), SYMBOL_BATCH_SIZE):
        batch_ids = asset_ids[i : i + SYMBOL_BATCH_SIZE]
        batch_symbols = [active_assets[bid] for bid in batch_ids]

        # A. Determine Earliest Needed Start for this Batch
        batch_needed_start = end_dt
        for bid in batch_ids:
            last = latest_map.get(bid)
            # If we have data, start 1 interval after the last bar
            if last:
                # Add 1 unit of time (e.g. 5 mins) to avoid re-fetching last bar
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

        # B. Calculate Max Days per Request (The "Chunk Size")
        # Formula: (Max_Limit * Safety) / (Num_Symbols * Bars_Per_Day)
        # This dynamically adapts. If you switch to 1-min bars, the time chunk shrinks automatically.
        target_datapoints = settings.ALPACA_MAX_DATAPOINTS * settings.API_SAFETY_RATIO
        daily_load_per_batch = len(batch_symbols) * bars_per_day_estimate

        # Avoid division by zero
        if daily_load_per_batch == 0:
            daily_load_per_batch = 1

        max_trading_days_per_req = math.floor(target_datapoints / daily_load_per_batch)
        # Clamp: At least 1 day, max 30 days (to prevent timeouts on huge empty ranges)
        chunk_days_limit = max(1, min(30, max_trading_days_per_req))

        print(
            f"Batch {i//SYMBOL_BATCH_SIZE + 1}: {len(batch_symbols)} syms. "
            f"Density: {daily_load_per_batch} bars/day. "
            f"calc_chunk_size: {chunk_days_limit} trading days."
        )

        # C. Time Chunk Loop (Using Trading Schedule)
        # We iterate through the schedule instead of adding random calendar days

        # Filter schedule for relevant range
        relevant_schedule = full_schedule[
            (full_schedule.index >= pd.Timestamp(batch_needed_start.date()))
            & (full_schedule.index <= pd.Timestamp(end_dt.date()))
        ]

        # If no trading days in range, just skip (e.g., it's Sunday)
        if relevant_schedule.empty:
            continue

        schedule_dates = relevant_schedule.index.to_list()

        # Iterate through schedule in chunks
        for j in range(0, len(schedule_dates), chunk_days_limit):
            chunk_dates = schedule_dates[j : j + chunk_days_limit]

            # Start of chunk (Market Open of first day)
            chunk_start_date = chunk_dates[0]
            # End of chunk (Market Close of last day) + buffer to cover after hours
            chunk_end_date = chunk_dates[-1] + timedelta(days=1)

            # Ensure dates are UTC *before* comparison
            chunk_start_utc = ensure_utc_timestamp(chunk_start_date)
            chunk_end_utc = ensure_utc_timestamp(chunk_end_date)

            # Convert to UTC datetime for API
            # Calculate Start
            req_start = batch_needed_start if j == 0 else chunk_start_utc

            # Calculate End (Now safe to compare with end_dt)
            req_end = min(chunk_end_utc, end_dt)

            if req_start >= req_end:
                continue

            print(f"  -> Fetching Chunk: {req_start.date()} to {req_end.date()}")

            df = source.get_ohlcv_bars(
                symbols=batch_symbols,
                start_dt=req_start,
                end_dt=req_end,
                timeframe=timeframe,
            )

            if not df.is_empty():
                count = await _write_to_db(df, active_assets, ModelClass)
                print(f"     -> Wrote {count} rows.")

            await asyncio.sleep(0.5)


async def _write_to_db(
    df: pl.DataFrame, active_assets: dict[int, str], model_class
) -> int:
    """
    Helper function to write OHLCV dataframes to the database.
    Handles: Joining Asset IDs, Selecting Columns, and Chunked Inserts.
    """
    if df.is_empty():
        return 0

    # 1. Map Symbol string back to Asset ID integer
    # active_assets is {id: symbol}, we need {symbol: id} for the join
    # Constructing a small DataFrame is the fastest way to join in Polars
    asset_ids = list(active_assets.keys())
    symbols = list(active_assets.values())

    map_df = pl.DataFrame({"symbol": symbols, "asset_id": asset_ids})

    # Perform the join
    df = df.join(map_df, on="symbol", how="left")

    # 2. Select and Clean Columns
    # We only want columns that exist in our database model
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

    # Ensure we only select columns that actually exist in the dataframe (safety check)
    available_cols = [c for c in target_cols if c in df.columns]

    # Convert to list of dictionaries for SQLAlchemy
    records = df.select(available_cols).to_dicts()

    if not records:
        return 0

    total_written = 0

    # 3. Chunked Write
    async with get_db_session() as session:
        chunk_size = settings.DB_WRITE_CHUNK_SIZE

        for start_idx in range(0, len(records), chunk_size):
            chunk = records[start_idx : start_idx + chunk_size]

            # Construct Insert Statement
            stmt = insert(model_class).values(chunk)

            # Handle Duplicates: Ignore if (time, asset_id) exists
            stmt = stmt.on_conflict_do_nothing(index_elements=["time", "asset_id"])

            await session.execute(stmt)
            await session.commit()

            total_written += len(chunk)

    return total_written
