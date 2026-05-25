use crate::Result;

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

/// Open a connection pool to Postgres.
///
/// A pool (rather than a single connection) lets many tasks run queries
/// concurrently, reusing a bounded set of connections instead of opening one per
/// query. `max_connections` caps that set — size it to what the database is
/// configured to accept.
pub async fn connect(database_url: &str, max_connections: u32) -> Result<PgPool> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(database_url)
        .await
}
