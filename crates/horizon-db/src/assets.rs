//! Read/write operations on the `assets` table — the universe of tradable
//! symbols and which of them are currently active.

use crate::types::{Asset, AssetId, NewAsset};
use crate::{Db, Result};

impl Db {
    /// Every active asset, ordered by symbol.
    ///
    /// "Active" is the screener's verdict (see [`Db::refresh_screener`]); this is
    /// the working universe the ingestion pipeline iterates over.
    pub async fn active_assets(&self) -> Result<Vec<Asset>> {
        // `query!` (checked macro): sqlx verifies these columns and their types
        // against the real table at compile time and generates the row struct.
        // `fetch_all` because we want every matching row.
        let rows = sqlx::query!(
            r#"
            SELECT id, symbol, name, exchange, asset_type, is_active
            FROM assets
            WHERE is_active = true
            ORDER BY symbol
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        // Translate raw rows into domain types at the boundary: the bare i32 id
        // becomes an AssetId so the rest of the code can't confuse it with some
        // other kind of id.
        Ok(rows
            .into_iter()
            .map(|r| Asset {
                id: AssetId::new(r.id),
                symbol: r.symbol,
                name: r.name,
                exchange: r.exchange,
                asset_type: r.asset_type,
                is_active: r.is_active,
            })
            .collect())
    }

    /// Insert new assets or update existing ones, marking each active.
    ///
    /// For ad-hoc additions; the screener refresh uses [`Db::refresh_screener`].
    pub async fn upsert_assets(&self, assets: &[NewAsset]) -> Result<u64> {
        upsert_assets_into(&self.pool, assets).await
    }

    /// Atomically replace the active universe with `fresh`.
    ///
    /// The screener runs as a "mark and sweep": deactivate everything, then
    /// re-activate exactly the symbols that passed this run's criteria. Symbols
    /// that dropped out keep their rows and price history — they just go
    /// inactive.
    ///
    /// Both statements run inside one transaction, and that matters: between the
    /// UPDATE and the upsert, *every* asset is inactive, so a reader calling
    /// `active_assets` in that window would see nothing. The transaction makes
    /// the swap atomic (readers see the old set or the new set, never the empty
    /// middle), and a failure partway through rolls back rather than leaving the
    /// whole universe deactivated.
    pub async fn refresh_screener(&self, fresh: &[NewAsset]) -> Result<u64> {
        // begin() opens a transaction; nothing here is visible to other
        // connections until commit().
        let mut tx = self.pool.begin().await?;

        // Sweep: everyone inactive to start.
        sqlx::query!("UPDATE assets SET is_active = false")
            .execute(&mut *tx)
            .await?;

        // Mark: re-activate (and insert/update) the fresh universe on the SAME
        // transaction. `&mut *tx` reborrows the transaction as the executor.
        let affected = upsert_assets_into(&mut *tx, fresh).await?;

        tx.commit().await?;
        Ok(affected)
    }
}

/// Shared UNNEST upsert for assets.
///
/// Generic over the executor `E` so the identical SQL runs either directly on a
/// pool (autocommit, from `upsert_assets`) or on a transaction (from
/// `refresh_screener`). Both `&PgPool` and `&mut PgConnection` — which is what
/// `&mut *tx` derefs to — implement `PgExecutor`, so one function serves both
/// callers without duplicating the query.
async fn upsert_assets_into<'e, E>(executor: E, assets: &[NewAsset]) -> Result<u64>
where
    E: sqlx::PgExecutor<'e>,
{
    if assets.is_empty() {
        return Ok(0);
    }

    // Struct-of-arrays for the UNNEST bulk insert (see `insert_bars` for why this
    // beats a multi-row VALUES list: one array param per column dodges the
    // 65535-bind-parameter ceiling). `as_deref()` turns `&Option<String>` into
    // `Option<&str>` so we bind borrowed text instead of cloning.
    let symbols: Vec<&str> = assets.iter().map(|a| a.symbol.as_str()).collect();
    let names: Vec<Option<&str>> = assets.iter().map(|a| a.name.as_deref()).collect();
    let exchanges: Vec<Option<&str>> = assets.iter().map(|a| a.exchange.as_deref()).collect();
    let types: Vec<&str> = assets.iter().map(|a| a.asset_type.as_str()).collect();

    // `query` (no `!`): runtime-checked, like the other UNNEST inserts.
    //
    // `ON CONFLICT (symbol) DO UPDATE` is the upsert: if the symbol already
    // exists, refresh its metadata from the incoming row (`EXCLUDED` is the row
    // we tried to insert) and set it active. Setting `is_active = true` here is
    // what makes the mark-and-sweep work — re-inserting a symbol flips it back on
    // after the blanket deactivate. Note this keys on `symbol`, so a *renamed*
    // ticker arrives as a new row; only company-name changes flow through
    // `EXCLUDED.name` automatically.
    let affected = sqlx::query(
        r#"
        INSERT INTO assets (symbol, name, exchange, asset_type, is_active)
        SELECT symbol, name, exchange, asset_type, true
        FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[])
            AS t(symbol, name, exchange, asset_type)
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            exchange = EXCLUDED.exchange,
            asset_type = EXCLUDED.asset_type,
            is_active = true
        "#,
    )
    .bind(&symbols)
    .bind(&names)
    .bind(&exchanges)
    .bind(&types)
    .execute(executor)
    .await?
    .rows_affected();

    Ok(affected)
}
