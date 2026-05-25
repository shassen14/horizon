//! `horizon-db` — the data-access layer for Horizon's market data.
//!
//! Everything goes through a single [`Db`] handle: it owns a connection pool,
//! and each table's operations live in a sibling module (`assets`,
//! `market_data`, `corporate_actions`, `pipelines_runs`) as inherent methods on
//! `Db`. We deliberately don't hide this behind a `Repository` trait — that
//! would throw away sqlx's compile-time query checking and invite mock-based
//! tests that pass while the real schema is broken. The domain types these
//! methods speak in live in [`types`].

mod assets;
mod corporate_actions;
mod market_data;
mod pipelines_runs;
mod pool;
pub mod types;

// Re-export the domain types at the crate root so callers write
// `horizon_db::AssetId` instead of `horizon_db::types::AssetId`.
pub use types::{
    ActionKind, Asset, AssetId, CorporateAction, DailyKey, NewAsset, NewBar, RunId, RunOutcome,
    Source,
};

use sqlx::PgPool;

/// Crate-wide result type. The error is `sqlx::Error` directly (rather than a
/// bespoke error enum) so callers keep full fidelity — they can tell a dropped
/// connection (worth retrying) from a constraint violation or decode error (a
/// bug to fix). Defining a one-type-parameter `Result<T>` alias like this is a
/// normal Rust idiom (cf. `std::io::Result`): within this crate it shadows the
/// two-parameter `std::result::Result`, and anyone who needs the std one can
/// still spell it out in full.
pub type Result<T> = std::result::Result<T, sqlx::Error>;

/// A handle to the database: a cloneable wrapper around a connection pool.
///
/// `PgPool` is itself reference-counted internally, so `#[derive(Clone)]` is
/// cheap — cloning a `Db` shares the same pool, which is exactly what we want
/// when handing it to concurrent ingestion tasks.
#[derive(Clone, Debug)]
pub struct Db {
    pool: PgPool,
}

impl Db {
    /// Wrap an existing pool. Handy in tests, where `#[sqlx::test]` hands us a
    /// pool bound to an ephemeral database.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Connect to `database_url` and build a handle in one step.
    pub async fn connect(database_url: &str, max_connections: u32) -> Result<Self> {
        Ok(Self::new(
            pool::connect(database_url, max_connections).await?,
        ))
    }
}
