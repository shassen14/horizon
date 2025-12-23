# apps/db_worker/auditor.py

from datetime import datetime, timedelta, timezone
import pandas as pd
from sqlalchemy import text, update
from packages.database.session import get_db_session
from packages.database.models import Asset
from packages.quant_lib.date_utils import (
    ensure_utc_timestamp,
    get_full_trading_schedule,
)


class DataAuditor:
    def __init__(self, logger):
        self.logger = logger
        # Cache the schedule to avoid re-calculating it 17,000 times
        self.schedule = None

    async def run_daily_audit(self):
        """Checks for gaps in Daily data larger than 4 days."""
        self.logger.info("--- Auditing Daily Data ---")
        await self._ensure_schedule()

        query = text(
            """
            WITH gaps AS (
                SELECT 
                    asset_id,
                    time,
                    LEAD(time) OVER (PARTITION BY asset_id ORDER BY time) AS next_time
                FROM market_data_daily
                WHERE time > NOW() - INTERVAL '2 years'
            )
            SELECT asset_id, time AS gap_start, next_time AS gap_end
            FROM gaps
            WHERE next_time - time > INTERVAL '4 days';
        """
        )

        await self._process_gaps(
            query, "last_market_data_daily_update", is_intraday=False
        )

    async def run_intraday_audit(self):
        """Checks for gaps in 5-min data larger than 24 hours."""
        self.logger.info("--- Auditing Intraday (5m) Data ---")
        await self._ensure_schedule()

        # Check last 60 days (checking 3 years takes too long for routine audits)
        query = text(
            """
            WITH gaps AS (
                SELECT 
                    asset_id,
                    time,
                    LEAD(time) OVER (PARTITION BY asset_id ORDER BY time) AS next_time
                FROM market_data_5min
                WHERE time > NOW() - INTERVAL '60 days'
            )
            SELECT asset_id, time AS gap_start, next_time AS gap_end
            FROM gaps
            WHERE next_time - time > INTERVAL '24 hours';
        """
        )

        await self._process_gaps(
            query, "last_market_data_5min_update", is_intraday=True
        )

    async def _ensure_schedule(self):
        if self.schedule is None:
            # Load schedule for last 5 years to cover all potential audit ranges
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=365 * 5)
            self.logger.info("Loading NYSE Trading Calendar...")
            self.schedule = get_full_trading_schedule(start.date(), end.date())

    def _is_real_gap(
        self, start_dt: datetime, end_dt: datetime, is_intraday: bool
    ) -> bool:
        """
        Determines if a time gap represents missing data or just market closure.
        """
        # Convert to UTC for comparison if not already
        start_dt = ensure_utc_timestamp(start_dt)
        end_dt = ensure_utc_timestamp(end_dt)

        # Get schedule slice between the two dates
        # We look for any rows in the schedule where the market was open
        # slice is inclusive on pandas indices

        # Optimization: Just check dates first
        relevant_days = self.schedule[
            (self.schedule.index >= pd.Timestamp(start_dt.date()))
            & (self.schedule.index <= pd.Timestamp(end_dt.date()))
        ]

        if relevant_days.empty:
            # No market days in this range -> False Positive (Weekend/Holiday)
            return False

        if not is_intraday:
            # For Daily data:
            # If we have > 1 row in schedule between start and end (exclusive), we missed a day.
            # Example: Data on Mon, Data on Fri. Gap is Tue, Wed, Thu.
            # Schedule has Tue, Wed, Thu. len > 0 -> Real Gap.

            # Logic: Count valid trading days strictly BETWEEN start and end
            mask = (relevant_days.index.date > start_dt.date()) & (
                relevant_days.index.date < end_dt.date()
            )
            missed_days = len(relevant_days[mask])
            return missed_days > 0

        else:
            # For Intraday data:
            # We check if there is any overlap between [Start, End] and [Market Open, Market Close]
            # BUT we exclude the exact start/end points (since data exists there).

            for index, row in relevant_days.iterrows():
                # Get Market Open/Close for this specific day (handles Early Closes!)
                mkt_open = row["market_open"]
                mkt_close = row["market_close"]

                # Ensure UTC
                if mkt_open.tzinfo is None:
                    mkt_open = mkt_open.replace(tzinfo=timezone.utc)
                if mkt_close.tzinfo is None:
                    mkt_close = mkt_close.replace(tzinfo=timezone.utc)
                else:
                    mkt_open = mkt_open.astimezone(timezone.utc)
                    mkt_close = mkt_close.astimezone(timezone.utc)

                # Check for overlap
                # Effectively: Did the market exist between our gap?
                # Gap: 4pm Wed -> 9:30am Fri (Thanksgiving).
                # Thursday: Closed (No row in schedule).
                # Friday row: Open 9:30.
                # Overlap: max(gap_start, mkt_open) < min(gap_end, mkt_close)

                overlap_start = max(start_dt, mkt_open)
                overlap_end = min(end_dt, mkt_close)

                # If there is valid time between the overlap, we missed data.
                # We add a small buffer (e.g. 10 mins) to allow for slight timestamp misalignment
                if (overlap_end - overlap_start) > timedelta(minutes=10):
                    return True

            return False

    async def _process_gaps(self, query, ledger_column, is_intraday):
        async with get_db_session() as session:
            result = await session.execute(query)
            raw_gaps = result.fetchall()

            if not raw_gaps:
                self.logger.success("No raw time gaps found.")
                return

            self.logger.info(
                f"Analyzing {len(raw_gaps)} raw gaps for market validity..."
            )

            rewind_map = {}
            false_positives = 0

            # Verify each gap against the calendar
            for row in raw_gaps:
                asset_id = row[0]
                gap_start = row[1]
                gap_end = row[2]

                if self._is_real_gap(gap_start, gap_end, is_intraday):
                    # It's a real gap. Track the earliest rewind point.
                    if asset_id not in rewind_map or gap_start < rewind_map[asset_id]:
                        rewind_map[asset_id] = gap_start
                else:
                    false_positives += 1

            self.logger.info(
                f"Analysis complete. {false_positives} false positives ignored (Holidays/Weekends)."
            )

            if not rewind_map:
                self.logger.success("All gaps were false positives. No repair needed.")
                return

            self.logger.warning(
                f"Repairing ledgers for {len(rewind_map)} assets with REAL missing data..."
            )

            # Batch Update Logic
            count = 0
            chunk_size = 100
            col_attr = getattr(Asset, ledger_column)
            items = list(rewind_map.items())

            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                for asset_id, earliest_gap in chunk:
                    stmt = (
                        update(Asset)
                        .where((Asset.id == asset_id) & (col_attr > earliest_gap))
                        .values({col_attr: earliest_gap})
                    )
                    res = await session.execute(stmt)
                    if res.rowcount > 0:
                        count += 1
                await session.commit()

            self.logger.success(f"Audit Complete. Rewound ledgers for {count} assets.")
