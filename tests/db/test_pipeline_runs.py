import datetime

import pytest
from psycopg_pool import AsyncConnectionPool

from horizon.db.pipeline_runs import finish_run, last_successful_run, start_run
from horizon.db.types import Failed, RunId, RunOutcome, Skipped, Success

JOB = "ingest_daily"


async def fetch_run(
    db: AsyncConnectionPool, run_id: RunId
) -> tuple[str, datetime.datetime | None, int, str | None]:
    async with db.connection() as conn:
        cur = await conn.execute(
            """
            SELECT status, finished_at, rows_processed, error
            FROM pipeline_runs
            WHERE id = %s
            """,
            (run_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        return row


@pytest.mark.asyncio
async def test_start_run_opens_a_running_row(db: AsyncConnectionPool) -> None:
    first = await start_run(db, JOB)
    second = await start_run(db, JOB)

    # Each run gets its own id, even for the same job.
    assert first != second
    assert await fetch_run(db, first) == ("running", None, 0, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "status", "error"),
    [
        (Success(), "success", None),
        (Failed(message="alpaca timed out"), "failed", "alpaca timed out"),
        (Skipped(), "skipped", None),
    ],
)
async def test_finish_run_records_outcome(
    db: AsyncConnectionPool, outcome: RunOutcome, status: str, error: str | None
) -> None:
    run_id = await start_run(db, JOB)

    await finish_run(db, run_id, outcome, rows_processed=42)

    stored_status, finished_at, rows_processed, stored_error = await fetch_run(
        db, run_id
    )
    assert (stored_status, rows_processed, stored_error) == (status, 42, error)
    assert finished_at is not None


@pytest.mark.asyncio
async def test_finish_run_cannot_overwrite_a_finished_run(
    db: AsyncConnectionPool,
) -> None:
    run_id = await start_run(db, JOB)
    await finish_run(db, run_id, Success(), rows_processed=42)

    # Only a running row can be finished, so a second call leaves the first
    # result in place.
    await finish_run(db, run_id, Failed(message="late retry"), rows_processed=0)

    status, _, rows_processed, error = await fetch_run(db, run_id)
    assert (status, rows_processed, error) == ("success", 42, None)


@pytest.mark.asyncio
async def test_last_successful_run_without_success_is_none(
    db: AsyncConnectionPool,
) -> None:
    assert await last_successful_run(db, JOB) is None

    # Running and failed runs don't count as successes.
    await start_run(db, JOB)
    failed = await start_run(db, JOB)
    await finish_run(db, failed, Failed(message="boom"), rows_processed=0)

    assert await last_successful_run(db, JOB) is None


@pytest.mark.asyncio
async def test_last_successful_run_is_latest_finish(db: AsyncConnectionPool) -> None:
    # The earlier-started run finishes last, so ordering by start time would
    # return the wrong run.
    slow = await start_run(db, JOB)
    fast = await start_run(db, JOB)
    await finish_run(db, fast, Success(), rows_processed=1)
    await finish_run(db, slow, Success(), rows_processed=1)

    # Another job's success must not leak into this job's answer.
    other = await start_run(db, "features_daily")
    await finish_run(db, other, Success(), rows_processed=1)

    _, slow_finished_at, _, _ = await fetch_run(db, slow)
    result = await last_successful_run(db, JOB)

    assert result == slow_finished_at
    assert result is not None
    assert result.tzinfo == datetime.UTC
