//! Read/write operations on the `market_data_daily` hypertable.

use crate::types::{AssetId, DailyKey, NewBar};
use crate::{Db, Result};

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

impl Db {
    /// For each requested asset, the timestamp of its most recent bar.
    ///
    /// This is the *incremental cursor* for ingestion: "I already have data up
    /// to here, so only fetch newer bars." Assets with no bars yet simply don't
    /// appear in the map, and the caller treats a missing entry as "start from
    /// the beginning."
    ///
    /// The caller passes whichever assets it cares about (normally the active
    /// universe); this function stays policy-free and just answers the question
    /// for whatever ids it's handed.
    pub async fn latest_times(&self, asset_ids: &[AssetId]) -> Result<HashMap<AssetId, DailyKey>> {
        // The query binds a single Postgres array, so flatten our newtypes down
        // to the raw i32 the driver knows how to send.
        let ids: Vec<i32> = asset_ids.iter().map(|x| x.get()).collect();

        // `query!` (with the `!`) is the *compile-time-checked* macro: at build
        // time sqlx sends this SQL to the database, learns the real column types,
        // and generates the result struct (giving us `r.asset_id`, `r.max_time`).
        // If the SQL is wrong or a column type changes, the build fails instead
        // of surprising us at runtime.
        //
        // `r#"..."#` is a raw string literal — inside it, quotes and backslashes
        // are literal, so the SQL reads naturally without escaping.
        //
        // The SQL: the latest bar time per asset.
        //   - `WHERE asset_id = ANY($1)` matches against the array bound as `$1`
        //     (Postgres positional parameter #1). `ANY(array)` is the clean way
        //     to say "asset_id IN (this whole list)" with a single bound param.
        //   - `GROUP BY asset_id` + `MAX(time)` yields one row per asset.
        //   - `AS "max_time!"`: `max_time` is the column alias (it becomes the
        //     `r.max_time` field). The trailing `!` is a sqlx annotation, not
        //     SQL — it forces the column to NON-NULL. sqlx assumes `MAX(...)`
        //     might be NULL (an aggregate over zero rows is), but `GROUP BY`
        //     guarantees each group has a row, so we override to a plain
        //     `DateTime<Utc>` instead of `Option<_>`. The quotes are required
        //     because `!` isn't a legal bare-identifier character.
        let rows = sqlx::query!(
            r#"
            SELECT asset_id, MAX(time) AS "max_time!"
            FROM market_data_daily
            WHERE asset_id = ANY($1)
            GROUP BY asset_id
            "#,
            &ids
        )
        // `fetch_all` because we want every group's row as a `Vec`. (Siblings:
        // `fetch_one` for exactly one row, `fetch_optional` for zero-or-one.)
        .fetch_all(&self.pool)
        .await?;

        // Translate raw rows back into domain types at the boundary.
        // `from_db_unchecked` skips re-validating the midnight-UTC invariant —
        // the DB's CHECK already guaranteed it on the way in.
        Ok(rows
            .into_iter()
            .map(|r| {
                (
                    AssetId::new(r.asset_id),
                    DailyKey::from_db_unchecked(r.max_time),
                )
            })
            .collect())
    }

    /// Bulk-insert daily bars, skipping any that already exist.
    ///
    /// Idempotent by design: re-running ingestion for a day that's already
    /// stored is a no-op, so a crashed-and-retried job can't create duplicates
    /// or error out.
    pub async fn insert_bars(&self, bars: &[NewBar]) -> Result<u64> {
        if bars.is_empty() {
            return Ok(0);
        }

        // Struct-of-arrays: instead of one row per bar, transpose into one array
        // per column. That's what the UNNEST insert below consumes.
        let times: Vec<DateTime<Utc>> = bars.iter().map(|b| b.time.as_utc()).collect();
        let asset_ids: Vec<i32> = bars.iter().map(|b| b.asset_id.get()).collect();
        let opens: Vec<Decimal> = bars.iter().map(|b| b.open).collect();
        let highs: Vec<Decimal> = bars.iter().map(|b| b.high).collect();
        let lows: Vec<Decimal> = bars.iter().map(|b| b.low).collect();
        let closes: Vec<Decimal> = bars.iter().map(|b| b.close).collect();
        let volumes: Vec<i64> = bars.iter().map(|b| b.volume).collect();
        let vwaps: Vec<Option<Decimal>> = bars.iter().map(|b| b.vwap).collect();
        let sources: Vec<&str> = bars.iter().map(|b| b.source.as_str()).collect();

        // This uses `query` (no `!`), the *runtime*-checked builder, not the
        // macro. Two reasons:
        //   1. The macro's type inference doesn't handle the array-of-Decimal
        //      UNNEST binds cleanly, so it fights us here.
        //   2. We lose little: the idempotency test covers this insert.
        // It is NOT unsafe — values still travel as bound parameters ($1..$9),
        // never string-concatenated, so there's no SQL-injection surface.
        //
        // Why UNNEST instead of a normal multi-row INSERT? Postgres caps a single
        // statement at 65535 bind parameters. A `VALUES (...),(...)` insert binds
        // one parameter *per cell* — 10 columns x N rows — so it tops out around
        // ~6500 rows. UNNEST instead takes one *array* per column (9 params
        // total, regardless of row count) and expands them row-wise into a
        // virtual table we SELECT from; the whole batch goes in one round trip.
        // `'none'` is hard-coded for the adjustment column because we only ever
        // store raw bars.
        //
        // `ON CONFLICT (time, asset_id) DO NOTHING` is what makes it idempotent:
        // a bar already present (same primary key) is silently skipped.
        let inserted = sqlx::query(
            r#"
            INSERT INTO market_data_daily (time, asset_id, open, high, low, close, volume, vwap, source, adjustment)
            SELECT time, asset_id, open, high, low, close, volume, vwap, source, 'none'
            FROM UNNEST(
                $1::timestamptz[], $2::int4[], $3::numeric[], $4::numeric[], $5::numeric[], $6::numeric[], $7::int8[], $8::numeric[], $9::text[])
            AS t(time, asset_id, open, high, low, close, volume, vwap, source)
            ON CONFLICT (time, asset_id) DO NOTHING
            "#,
        )
        // .bind order matches $1..$9 above; each Vec is sent as one array param.
        .bind(&times)
        .bind(&asset_ids)
        .bind(&opens)
        .bind(&highs)
        .bind(&lows)
        .bind(&closes)
        .bind(&volumes)
        .bind(&vwaps)
        .bind(&sources)
        .execute(&self.pool)
        .await?
        // rows_affected counts only rows actually inserted, so conflicts that
        // were skipped don't count — that's our "how many were new?" answer.
        .rows_affected();

        Ok(inserted)
    }
}
