//! Integration tests for the database layer.
//!
//! Each test is annotated with `#[sqlx::test(migrations = "../../migrations")]`,
//! which (using the `DATABASE_URL` env var) creates a fresh, empty database,
//! runs every migration in `migrations/` against it, hands the test a pool bound
//! to that database, and drops it afterward. So tests are fully isolated from
//! each other and from real data — at the cost of running all migrations per
//! test, which is why each takes a few seconds. The path is relative to this
//! crate's root, so `../../migrations` points at the repo-root `migrations/`.

use chrono::{DateTime, NaiveDate, Utc};
use horizon_db::{
    ActionKind, AssetId, CorporateAction, DailyKey, Db, NewAsset, NewBar, RunOutcome, Source,
};
use rust_decimal::Decimal;
use sqlx::PgPool;

fn px(s: &str) -> Decimal {
    s.parse().unwrap()
}

fn na(symbol: &str) -> NewAsset {
    NewAsset {
        symbol: symbol.into(),
        name: Some(format!("{symbol} Inc")),
        exchange: Some("XNAS".into()),
        asset_type: "stock".into(),
    }
}

fn bar(asset_id: AssetId, date: &str) -> NewBar {
    NewBar {
        time: DailyKey::from_trading_date(date.parse::<NaiveDate>().unwrap()),
        asset_id,
        open: px("100.00"),
        high: px("105.00"),
        low: px("99.00"),
        close: px("104.00"),
        volume: 1_000_000,
        vwap: Some(px("102.00")),
        source: Source::Alpaca,
    }
}

async fn seed_asset(db: &Db, symbol: &str) -> AssetId {
    db.upsert_assets(&[na(symbol)]).await.unwrap();
    db.active_assets()
        .await
        .unwrap()
        .into_iter()
        .find(|a| a.symbol == symbol)
        .unwrap()
        .id
}

#[sqlx::test(migrations = "../../migrations")]
async fn insert_bars_is_idempotent(pool: PgPool) {
    let db = Db::new(pool);
    let id = seed_asset(&db, "AAPL").await;
    let bars = vec![bar(id, "2024-01-02"), bar(id, "2024-01-03")];

    assert_eq!(db.insert_bars(&bars).await.unwrap(), 2);
    // ON CONFLICT (time, asset_id) DO NOTHING -> second run inserts nothing
    assert_eq!(db.insert_bars(&bars).await.unwrap(), 0);
}

#[sqlx::test(migrations = "../../migrations")]
async fn latest_times_returns_max_per_asset(pool: PgPool) {
    let db = Db::new(pool);
    let id = seed_asset(&db, "AAPL").await;
    db.insert_bars(&[bar(id, "2024-01-02"), bar(id, "2024-01-05"), bar(id, "2024-01-03")])
        .await
        .unwrap();

    let map = db.latest_times(&[id]).await.unwrap();
    assert_eq!(
        map[&id],
        DailyKey::from_trading_date("2024-01-05".parse().unwrap())
    );
}

#[sqlx::test(migrations = "../../migrations")]
async fn rejects_non_midnight_bar(pool: PgPool) {
    // The UTC-midnight CHECK is an invariant DailyKey can't violate, so test
    // it with a raw insert that bypasses the type.
    let db = Db::new(pool.clone());
    let id = seed_asset(&db, "AAPL").await;

    let res = sqlx::query(
        "INSERT INTO market_data_daily
            (time, asset_id, open, high, low, close, volume, source, adjustment)
         VALUES ($1, $2, 1, 1, 1, 1, 1, 'manual', 'none')",
    )
    .bind("2024-01-02T09:30:00Z".parse::<DateTime<Utc>>().unwrap())
    .bind(id.get())
    .execute(&pool)
    .await;

    assert!(res.is_err(), "non-midnight time must violate the CHECK");
}

#[sqlx::test(migrations = "../../migrations")]
async fn upsert_updates_name_and_reactivates(pool: PgPool) {
    let db = Db::new(pool);
    let id = seed_asset(&db, "FB").await;

    // company name change on the same symbol flows through EXCLUDED.name
    db.upsert_assets(&[NewAsset {
        symbol: "FB".into(),
        name: Some("Meta (was Facebook)".into()),
        exchange: Some("XNAS".into()),
        asset_type: "stock".into(),
    }])
    .await
    .unwrap();

    let a = db
        .active_assets()
        .await
        .unwrap()
        .into_iter()
        .find(|a| a.id == id)
        .unwrap();
    assert_eq!(a.name.as_deref(), Some("Meta (was Facebook)"));
    assert!(a.is_active);
}

#[sqlx::test(migrations = "../../migrations")]
async fn refresh_screener_sweeps_inactive(pool: PgPool) {
    let db = Db::new(pool.clone());
    db.upsert_assets(&[na("AAPL"), na("TSLA"), na("OLD")])
        .await
        .unwrap();

    // new universe drops OLD, adds NVDA
    let n = db
        .refresh_screener(&[na("AAPL"), na("TSLA"), na("NVDA")])
        .await
        .unwrap();
    assert_eq!(n, 3);

    let active: Vec<String> = db
        .active_assets()
        .await
        .unwrap()
        .into_iter()
        .map(|a| a.symbol)
        .collect();
    assert_eq!(active, vec!["AAPL", "NVDA", "TSLA"]); // ordered by symbol, OLD gone

    // OLD is still in the table (history preserved), just inactive
    let old_active: bool = sqlx::query_scalar("SELECT is_active FROM assets WHERE symbol = 'OLD'")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(!old_active);
}

#[sqlx::test(migrations = "../../migrations")]
async fn upsert_actions_idempotent(pool: PgPool) {
    let db = Db::new(pool);
    let id = seed_asset(&db, "AAPL").await;
    let actions = vec![
        CorporateAction {
            asset_id: id,
            ex_date: "2024-06-10".parse().unwrap(),
            source: Source::Alpaca,
            kind: ActionKind::Split { ratio: px("4") },
        },
        CorporateAction {
            asset_id: id,
            ex_date: "2024-08-15".parse().unwrap(),
            source: Source::Alpaca,
            kind: ActionKind::Dividend { cash: px("0.24") },
        },
    ];

    assert_eq!(db.upsert_actions(&actions).await.unwrap(), 2);
    assert_eq!(db.upsert_actions(&actions).await.unwrap(), 0); // DO NOTHING
}

#[sqlx::test(migrations = "../../migrations")]
async fn corporate_action_check_rejects_mismatched_payload(pool: PgPool) {
    // A split must have a ratio and NO cash_amount; verify the CHECK holds.
    let db = Db::new(pool.clone());
    let id = seed_asset(&db, "AAPL").await;

    let res = sqlx::query(
        "INSERT INTO corporate_actions
            (asset_id, ex_date, action_type, split_ratio, cash_amount, source)
         VALUES ($1, '2024-06-10', 'split', 4, 0.24, 'alpaca')",
    )
    .bind(id.get())
    .execute(&pool)
    .await;

    assert!(res.is_err(), "split with cash_amount must violate the CHECK");
}

#[sqlx::test(migrations = "../../migrations")]
async fn pipeline_run_lifecycle(pool: PgPool) {
    let db = Db::new(pool);

    assert!(db.last_successful_run("daily_ingest").await.unwrap().is_none());

    let run = db.start_run("daily_ingest").await.unwrap();
    // running != success
    assert!(db.last_successful_run("daily_ingest").await.unwrap().is_none());

    db.finish_run(run, &RunOutcome::Success, 42).await.unwrap();
    assert!(db.last_successful_run("daily_ingest").await.unwrap().is_some());

    // a failed run doesn't overwrite the last success
    let bad = db.start_run("daily_ingest").await.unwrap();
    db.finish_run(bad, &RunOutcome::Failed("boom".into()), 0)
        .await
        .unwrap();
    assert!(db.last_successful_run("daily_ingest").await.unwrap().is_some());
}
