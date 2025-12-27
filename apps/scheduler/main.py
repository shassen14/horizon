# apps/scheduler/main.py

import asyncio
from datetime import datetime, timedelta, timezone

from packages.quant_lib.logging import LogManager
from packages.quant_lib.market_clock import MarketClock
from jobs import JobRunner

# --- 1. Setup (Composition Root) ---
# Create all the core objects once at startup.
log_manager = LogManager("scheduler")
logger = log_manager.get_logger("main")
jobs = JobRunner(log_manager)
clock = MarketClock()

# --- 2. State Management ---
# Use locks to prevent jobs from running on top of each other.
locks = {
    "ingest_intraday": asyncio.Lock(),
    "daily_routine": asyncio.Lock(),
    "data_audit": asyncio.Lock(),
    "db_optimize": asyncio.Lock(),
    "db_backup": asyncio.Lock(),
}

# Use state variables to ensure jobs only run once per period.
last_run_state = {
    "daily_routine": None,  # Will store date object
    "data_audit": None,
    "db_optimize": None,  # Will store day of week
    "db_backup": None,
}


# --- 3. The Main Loop ---
async def main():
    logger.info("--- Horizon Scheduler Started: Market-Aware Mode ---")

    while True:
        now = datetime.now(timezone.utc)

        # --- High-Frequency Job: Intraday Ingestion ---
        # Use the new, more accurate window check
        if clock.is_ingestion_window_open() and now.minute % 5 == 0:
            logger.info(f"Ingestion window is OPEN. Triggering Intraday job.")
            await _trigger_job_if_not_locked(
                "ingest_intraday", jobs.run_ingest_intraday
            )

        # --- Daily After-Market Job ---
        _, market_close = clock.get_session_times(now.date())

        if market_close:
            # We want to run 45 minutes AFTER the official close.
            daily_run_time = market_close + timedelta(minutes=45)

            if now >= daily_run_time and last_run_state["daily_routine"] != now.date():
                logger.info("Market has settled. Triggering Daily Routine.")
                await _trigger_job_if_not_locked(
                    "daily_routine", jobs.run_daily_routine
                )
                last_run_state["daily_routine"] = now.date()

        # --- Nightly Maintenance Job ---
        is_late_night = now.hour == 6  # ~1AM PST / 4AM EST
        if is_late_night and last_run_state["data_audit"] != now.date():
            await _trigger_job_if_not_locked("data_audit", jobs.run_data_audit)
            last_run_state["data_audit"] = now.date()

        # --- Weekly Maintenance Jobs (Run on Sunday) ---
        is_sunday = now.weekday() == 6
        if is_sunday and is_late_night:
            # DB Optimization
            if last_run_state["db_optimize"] != now.day:
                await _trigger_job_if_not_locked("db_optimize", jobs.run_db_optimize)
                last_run_state["db_optimize"] = now.day

            # DB Backup (Slightly later to not conflict)
            # if now.hour == 7 and last_run_state["db_backup"] != now.day:
            #     await _trigger_sync_job_if_not_locked("db_backup", jobs.run_db_backup)
            #     last_run_state["db_backup"] = now.day

        # Check every 60 seconds
        await asyncio.sleep(60)


# --- 4. Lock-Aware Trigger Helpers ---
async def _trigger_job_if_not_locked(job_name: str, coro):
    """Wrapper to run an async job with logging and lock protection."""
    lock = locks[job_name]

    if lock.locked():
        logger.warning(f"Job '{job_name}' skipped: Previous run is still active.")
        return

    async def job_wrapper():
        async with lock:
            await coro()

    asyncio.create_task(job_wrapper())


async def _trigger_sync_job_if_not_locked(job_name: str, func):
    """Wrapper for synchronous (blocking) jobs, running them in a thread."""
    lock = locks[job_name]

    if lock.locked():
        logger.warning(f"Job '{job_name}' skipped: Previous run is still active.")
        return

    async def job_wrapper():
        async with lock:
            logger.info(f">>> Starting SYNC job: {job_name}...")
            try:
                # Runs the blocking 'func' in a separate thread pool
                # and awaits its completion without freezing the main loop.
                await asyncio.to_thread(func)
                logger.success(f"<<< SYNC job: {job_name} complete.")
            except Exception as e:
                logger.exception(f"SYNC job '{job_name}' failed.")

    asyncio.create_task(job_wrapper())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down.")
