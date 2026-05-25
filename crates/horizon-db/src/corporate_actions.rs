//! Read/write operations on the `corporate_actions` table.

use crate::types::{ActionKind, CorporateAction};
use crate::{Db, Result};

use chrono::NaiveDate;
use rust_decimal::Decimal;

impl Db {
    /// Bulk-insert splits and dividends, skipping any already recorded.
    ///
    /// These rows are the raw material for adjust-on-read: prices are stored
    /// unadjusted, and we later replay these actions to compute split/dividend
    /// adjustment factors. An action for a given `(asset, ex_date, type)` never
    /// changes once it has happened, so re-ingesting is a no-op — hence the
    /// `ON CONFLICT ... DO NOTHING` and the idempotent return count.
    pub async fn upsert_actions(&self, actions: &[CorporateAction]) -> Result<u64> {
        if actions.is_empty() {
            return Ok(0);
        }

        // Same struct-of-arrays UNNEST strategy as `insert_bars` (one array per
        // column, sidestepping Postgres's 65535-bind-parameter ceiling) and the
        // same runtime `query` rather than the `query!` macro, because the macro
        // doesn't infer the array-of-Decimal binds cleanly.
        let asset_ids: Vec<i32> = actions.iter().map(|a| a.asset_id.get()).collect();
        let ex_dates: Vec<NaiveDate> = actions.iter().map(|a| a.ex_date).collect();
        let sources: Vec<&str> = actions.iter().map(|a| a.source.as_str()).collect();

        // Flatten the `ActionKind` sum type into the table's flat columns. This
        // match is the single place that translation lives: a Split contributes
        // a ratio and a NULL cash amount, a Dividend the reverse. The DB has a
        // CHECK constraint enforcing exactly that shape, so a mismatched row
        // (e.g. a split carrying a cash amount) would be rejected.
        let action_types: Vec<&str> = actions
            .iter()
            .map(|a| match a.kind {
                ActionKind::Split { .. } => "split",
                ActionKind::Dividend { .. } => "dividend",
            })
            .collect();
        let split_ratios: Vec<Option<Decimal>> = actions
            .iter()
            .map(|a| match a.kind {
                ActionKind::Split { ratio } => Some(ratio),
                ActionKind::Dividend { .. } => None,
            })
            .collect();
        let cash_amounts: Vec<Option<Decimal>> = actions
            .iter()
            .map(|a| match a.kind {
                ActionKind::Dividend { cash } => Some(cash),
                ActionKind::Split { .. } => None,
            })
            .collect();

        // SELECT from UNNEST(...) turns the six parallel arrays back into rows;
        // ON CONFLICT on the (asset_id, ex_date, action_type) primary key skips
        // anything already stored.
        let affected = sqlx::query(
            r#"
            INSERT INTO corporate_actions
                (asset_id, ex_date, action_type, split_ratio, cash_amount, source)
            SELECT asset_id, ex_date, action_type, split_ratio, cash_amount, source
            FROM UNNEST(
                $1::int4[], $2::date[], $3::text[], $4::numeric[], $5::numeric[], $6::text[]
            ) AS t(asset_id, ex_date, action_type, split_ratio, cash_amount, source)
            ON CONFLICT (asset_id, ex_date, action_type) DO NOTHING
            "#,
        )
        .bind(&asset_ids)
        .bind(&ex_dates)
        .bind(&action_types)
        .bind(&split_ratios)
        .bind(&cash_amounts)
        .bind(&sources)
        .execute(&self.pool)
        .await?
        .rows_affected();

        Ok(affected)
    }
}
