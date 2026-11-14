import datetime
from typing import assert_never

from psycopg_pool import AsyncConnectionPool

from horizon.db.types import Failed, RunId, RunOutcome, Skipped, Success


async def start_run(pool: AsyncConnectionPool, job_name: str) -> RunId:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO pipeline_runs (job_name, status)
            VALUES (%s, 'running')
            RETURNING id
            """,
            (job_name,),
        )

        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("INSERT ... RETURNING id returned no row")

    return RunId(row[0])


async def finish_run(
    pool: AsyncConnectionPool, run_id: RunId, outcome: RunOutcome, rows_processed: int
) -> None:
    match outcome:
        case Success():
            status, error = "success", None
        case Failed(message=message):
            status, error = "failed", message
        case Skipped():
            status, error = "skipped", None
        case _:
            assert_never(outcome)

    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE pipeline_runs
            SET status = %s, finished_at = now(), rows_processed = %s, error = %s
            WHERE id = %s AND status = 'running'
            """,
            (status, rows_processed, error, run_id),
        )


async def last_successful_run(
    pool: AsyncConnectionPool, job_name: str
) -> datetime.datetime | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT finished_at
            FROM pipeline_runs
            WHERE job_name = %s AND status = 'success'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (job_name,),
        )

        row = await cur.fetchone()
        if row is None:
            return None

    return row[0].astimezone(datetime.UTC)
