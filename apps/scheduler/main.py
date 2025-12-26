# apps/scheduler/main.py
import asyncio
import threading
from datetime import datetime, timezone
from packages.quant_lib.logging import LogManager
from market_clock import MarketClock
from jobs import JobRunner

# --- Setup ---
log_manager = LogManager("scheduler")
logger = log_manager.get_logger("main")
jobs = JobRunner(log_manager)
clock = MarketClock()

# --- Locks ---
ingest_intraday_lock = asyncio.Lock()
daily_routine_lock = asyncio.Lock()
audit_lock = asyncio.Lock()
optimize_lock = asyncio.Lock()
backup_lock = threading.Lock()  # For sync function


async def main():
    last_daily_run_date = None
    last_audit_run_date = None
    last_optimize_run_day = None  # Sunday = 6
    last_backup_run_day = None

    logger.info("--- Horizon Scheduler Started: Full Maintenance Mode ---")

    while True:
        now = datetime.now(timezone.utc)

        # 1. Intraday Ingestion (High Frequency)
        if (
            clock.is_market_open()
            and now.minute % 5 == 0
            and not ingest_intraday_lock.locked()
        ):
            asyncio.create_task(
                run_with_lock(ingest_intraday_lock, jobs.run_ingest_intraday)
            )

        # 2. Daily Ingestion + Features (After-Market)
        is_after_close = now.hour >= 21 and now.minute >= 15
        if (
            clock.is_trading_day(now.date())
            and is_after_close
            and now.date() != last_daily_run_date
            and not daily_routine_lock.locked()
        ):
            last_daily_run_date = now.date()
            asyncio.create_task(
                run_with_lock(daily_routine_lock, jobs.run_daily_routine)
            )

        # 3. Data Audit (Late Night)
        is_late_night = now.hour == 6  # 1 AM PST / 4 AM EST
        if (
            now.date() != last_audit_run_date
            and is_late_night
            and not audit_lock.locked()
        ):
            last_audit_run_date = now.date()
            asyncio.create_task(run_with_lock(audit_lock, jobs.run_data_audit))

        # 4. DB Optimization (Weekly - Sunday)
        is_sunday = now.weekday() == 6
        if (
            is_sunday
            and now.day != last_optimize_run_day
            and is_late_night
            and not optimize_lock.locked()
        ):
            last_optimize_run_day = now.day
            asyncio.create_task(run_with_lock(optimize_lock, jobs.run_db_optimize))

        # 5. DB Backup (Weekly - Sunday, slightly after)
        # is_late_sunday = is_sunday and now.hour == 7
        # if (
        #     is_late_sunday
        #     and now.day != last_backup_run_day
        #     and backup_lock.acquire(blocking=False)
        # ):
        #     last_backup_run_day = now.day
        #     # Run sync function in a separate thread to not block asyncio loop
        #     threading.Thread(
        #         target=run_sync_with_lock, args=(backup_lock, jobs.run_db_backup)
        #     ).start()

        await asyncio.sleep(60)


# --- Lock Helpers ---
async def run_with_lock(lock: asyncio.Lock, coro):
    async with lock:
        await coro()


def run_sync_with_lock(lock: threading.Lock, func):
    try:
        func()
    finally:
        lock.release()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down.")
