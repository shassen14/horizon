//! Read/write operations on `pipeline_runs` — the table that records every
//! ingestion/processing run for observability and scheduling.
//!
//! A run is opened with [`Db::start_run`] (status `'running'`) and closed with
//! [`Db::finish_run`] (terminal status + finish time). [`Db::last_successful_run`]
//! is the query the scheduler leans on as a dependency gate and freshness check.

use crate::types::{RunId, RunOutcome};
use crate::{Db, Result};

use chrono::{DateTime, Utc};

impl Db {
    /// Open a run row in the `'running'` state and return its id, which the
    /// caller later hands to [`Db::finish_run`].
    pub async fn start_run(&self, job_name: &str) -> Result<RunId> {
        // `query_scalar!` is like `query!` but for a query that returns a single
        // column — it hands back the value directly (here the new `id` as i64)
        // instead of a one-field row struct. `RETURNING id` is Postgres giving us
        // the IDENTITY value it just assigned, so we avoid a second round trip to
        // look it up.
        //
        // `fetch_one` because `INSERT ... RETURNING` produces exactly one row; it
        // errors if that somehow isn't the case.
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO pipeline_runs (job_name, status)
            VALUES ($1, 'running')
            RETURNING id
            "#,
            job_name
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(RunId::new(id))
    }

    /// Close a run: write its terminal status, finish time, rows processed, and
    /// (only for a failure) the error message.
    pub async fn finish_run(
        &self,
        run: RunId,
        outcome: &RunOutcome,
        rows_processed: i64,
    ) -> Result<()> {
        // `query!` (checked macro): the bound params are matched against the
        // column types at compile time. `now()` is evaluated by Postgres, so the
        // timestamp is the database's clock, not the app's. The enum maps to its
        // TEXT spellings via the `pub(crate)` helpers on `RunOutcome`, keeping
        // that persistence detail out of the call site. `execute` (not a
        // `fetch_*`) because an UPDATE returns no rows.
        sqlx::query!(
            r#"
            UPDATE pipeline_runs
            SET status = $2, finished_at = now(), rows_processed = $3, error = $4
            WHERE id = $1
            "#,
            run.get(),
            outcome.status(),
            rows_processed,
            outcome.error()
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// When did `job_name` last finish successfully? `None` means it never has.
    ///
    /// Drives the scheduler's dependency gate ("did the upstream job succeed
    /// before I run?") and the freshness check, without scanning the data
    /// tables.
    pub async fn last_successful_run(&self, job_name: &str) -> Result<Option<DateTime<Utc>>> {
        // ORDER BY finished_at DESC + LIMIT 1 = the most recent success.
        // `fetch_optional` because there may be zero matching rows (the job has
        // never succeeded), which gives us an `Option<row>`.
        let finished_at = sqlx::query_scalar!(
            r#"
            SELECT finished_at
            FROM pipeline_runs
            WHERE job_name = $1 AND status = 'success'
            ORDER BY finished_at DESC
            LIMIT 1
            "#,
            job_name
        )
        .fetch_optional(&self.pool)
        .await?;

        // Two layers of Option collapse here:
        //   - `fetch_optional` gives `Option<_>` (row present or not), and
        //   - `finished_at` is a nullable column, so the value itself is
        //     `Option<_>`.
        // So `finished_at` is `Option<Option<DateTime<Utc>>>`; `flatten()` turns
        // both "no row" and "row present but NULL" into `None`.
        Ok(finished_at.flatten())
    }
}
