//! Domain types for the database layer.
//!
//! These are the "vocabulary" of the crate: the newtypes and enums that the
//! query methods in the sibling modules (`assets`, `market_data`, ...) translate
//! to and from raw SQL rows. Keeping them in one module means there's a single
//! place to see the whole shape of the data we store.
//!
//! The guiding idea is "make illegal states unrepresentable": wrap bare ints in
//! newtypes so an `AssetId` can't be passed where a `RunId` is expected, and
//! model mutually-exclusive data (a corporate action is *either* a split *or* a
//! dividend) as an enum instead of a bag of nullable fields.

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use chrono_tz::America::New_York;
use rust_decimal::Decimal;

/// Primary key of a row in `assets`.
///
/// A newtype around `i32` (the column is `INTEGER GENERATED ALWAYS AS IDENTITY`)
/// rather than a bare `i32`, so the compiler stops us from mixing it up with,
/// say, a `RunId`. It's a *surrogate* key — the ticker symbol lives in its own
/// column — which is what lets a symbol change (FB -> META) without breaking the
/// link to that asset's historical price rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssetId(i32);

impl AssetId {
    pub fn new(id: i32) -> Self {
        AssetId(id)
    }

    /// Unwrap to the raw `i32`, e.g. to bind into a query.
    pub const fn get(&self) -> i32 {
        self.0
    }
}

/// The timestamp of a *daily* bar: an exchange trading date pinned to
/// 00:00:00 UTC.
///
/// Daily bars have no meaningful time-of-day, so we collapse each one to
/// midnight UTC and use that as half of the `(time, asset_id)` primary key. The
/// database enforces the same rule with a CHECK constraint; this type makes the
/// invariant hard to violate from Rust too, because every public constructor
/// produces midnight UTC by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DailyKey(DateTime<Utc>);

impl DailyKey {
    /// Build the key from a raw instant off a market-data feed (Alpaca, for
    /// instance, stamps a daily bar near the US session open in Eastern time).
    ///
    /// We can't just truncate the UTC instant to midnight: a bar stamped
    /// `2024-01-03T02:00:00Z` is still `2024-01-02` in New York, so truncating in
    /// UTC would file it under the wrong trading day. Instead we convert to the
    /// exchange's wall-clock timezone first, take the calendar date *there*, then
    /// pin that date to midnight UTC.
    pub fn from_et_instant(raw: DateTime<Utc>) -> Self {
        Self::from_trading_date(raw.with_timezone(&New_York).date_naive())
    }

    /// Build the key directly from a known trading date.
    ///
    /// `and_time(NaiveTime::MIN)` attaches midnight and `and_utc()` interprets it
    /// as UTC. Both are infallible, so there's no `.unwrap()`/`.expect()` and no
    /// way for this to panic — unlike `and_hms_opt(0, 0, 0)`, which returns an
    /// `Option` we'd have to unwrap.
    pub fn from_trading_date(date: NaiveDate) -> Self {
        Self(date.and_time(NaiveTime::MIN).and_utc())
    }

    /// Wrap a timestamp we just read back out of the database.
    ///
    /// `unchecked` because the DB's CHECK constraint already guaranteed it's
    /// midnight UTC on the way in — re-validating on every read would be wasted
    /// work. `pub(crate)` so only our own query code can call it; outside callers
    /// must go through the validating constructors above.
    pub(crate) fn from_db_unchecked(ts: DateTime<Utc>) -> Self {
        Self(ts)
    }

    /// The underlying instant, for binding into a query or doing date math.
    pub fn as_utc(&self) -> DateTime<Utc> {
        self.0
    }
}

/// Where a row came from. Stored as TEXT in the DB, but kept as a closed enum
/// here because *we* control the set of data providers — there's no need for an
/// open-ended string at the type level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Alpaca,
    Yfinance,
    Manual,
}

impl Source {
    /// The on-disk spelling. This is the single place the enum<->TEXT mapping
    /// lives, so the DB values and the Rust type can't drift apart.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Alpaca => "alpaca",
            Self::Yfinance => "yfinance",
            Self::Manual => "manual",
        }
    }
}

/// A daily OHLCV bar ready to be inserted.
///
/// Prices are `Decimal`, not `f64`: money needs exact base-10 arithmetic and
/// floats silently lose precision (`0.1 + 0.2 != 0.3`). The columns are
/// `NUMERIC(12,4)`, which `Decimal` maps to without rounding.
///
/// There's deliberately no `adjustment` field. We store bars *raw* (split- and
/// dividend-unadjusted) and adjust on read, so the column would always be
/// `'none'`; the insert hard-codes that rather than carrying a field that can
/// only hold one value. The column still exists in the table as a provenance
/// marker ("this row is raw"), which matters if we ever cache adjusted bars too.
pub struct NewBar {
    pub time: DailyKey,
    pub asset_id: AssetId,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: i64,
    pub vwap: Option<Decimal>,
    pub source: Source,
}

/// An asset row as it exists in the database — the *read* model.
///
/// Carries `id` (assigned by the DB) and `is_active` (server-controlled screener
/// state) because those only make sense for a row that already exists. Compare
/// with [`NewAsset`].
#[derive(Clone, Debug)]
pub struct Asset {
    pub id: AssetId,
    pub symbol: String,
    pub name: Option<String>,
    pub exchange: Option<String>,
    pub asset_type: String,
    pub is_active: bool,
}

/// The data we supply to *create or update* an asset — the *write* model.
///
/// No `id` (the DB mints it via IDENTITY) and no `is_active` (the upsert sets
/// it). Splitting this from [`Asset`] keeps "what I send" honest: we never have
/// to invent a fake id or unwrap an `Option<id>` for a row that doesn't exist
/// yet.
pub struct NewAsset {
    pub symbol: String,
    pub name: Option<String>,
    pub exchange: Option<String>,
    pub asset_type: String,
}

/// A split or dividend on a given ex-date.
///
/// These rows are the inputs to *adjust-on-read*: bars are stored raw, and when
/// we need adjusted prices we replay the actions to compute an adjustment
/// factor. Nothing consumes them at ingest time.
#[derive(Clone, Debug)]
pub struct CorporateAction {
    pub asset_id: AssetId,
    pub ex_date: NaiveDate,
    pub source: Source,
    pub kind: ActionKind,
}

/// The two kinds of action we track, as a sum type so each carries exactly the
/// payload it needs — a split has a ratio, a dividend has a cash amount, and
/// it's impossible to construct one with both or neither. The DB mirrors this
/// with a CHECK constraint (split => ratio set, cash null; and vice versa).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionKind {
    Split { ratio: Decimal },
    Dividend { cash: Decimal },
}

/// Primary key of a row in `pipeline_runs`. `i64` because the column is BIGINT —
/// runs accumulate forever and we don't want to wrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunId(i64);

impl RunId {
    pub const fn new(id: i64) -> Self {
        Self(id)
    }

    pub const fn get(&self) -> i64 {
        self.0
    }
}

/// How a pipeline run ended. `Failed` carries the error message so it can be
/// written to the `error` column for later debugging.
#[derive(Clone, Debug)]
pub enum RunOutcome {
    Success,
    Failed(String),
    Skipped,
}

impl RunOutcome {
    /// The `status` column value. `pub(crate)` — it's a detail of how we persist
    /// the enum, not part of the public API.
    pub(crate) fn status(&self) -> &'static str {
        match self {
            RunOutcome::Success => "success",
            RunOutcome::Failed(_) => "failed",
            RunOutcome::Skipped => "skipped",
        }
    }

    /// The `error` column value: only a `Failed` run has one.
    pub(crate) fn error(&self) -> Option<&str> {
        match self {
            RunOutcome::Failed(e) => Some(e.as_str()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // from_et_instant must bucket an instant by its *New York* calendar date,
    // not its UTC date — that's the whole reason the function exists, so the two
    // cases below pin the behavior (including a day where the two disagree).
    #[test]
    fn et_instant_picks_the_new_york_trading_date() {
        // 13:30 UTC = 09:30 in New York (EDT, summer) -> same calendar day.
        let market_open = "2024-07-01T13:30:00Z".parse::<DateTime<Utc>>().unwrap();
        assert_eq!(
            DailyKey::from_et_instant(market_open),
            DailyKey::from_trading_date(NaiveDate::from_ymd_opt(2024, 7, 1).unwrap()),
        );

        // 02:00 UTC on Jan 3 = 21:00 on Jan 2 in New York (EST, winter): the UTC
        // date and the trading date disagree, and the trading date is the one we
        // want.
        let late_night_utc = "2024-01-03T02:00:00Z".parse::<DateTime<Utc>>().unwrap();
        assert_eq!(
            DailyKey::from_et_instant(late_night_utc),
            DailyKey::from_trading_date(NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()),
        );
    }

    #[test]
    fn trading_date_is_pinned_to_midnight_utc() {
        let key = DailyKey::from_trading_date(NaiveDate::from_ymd_opt(2024, 1, 2).unwrap());
        assert_eq!(key.as_utc().to_rfc3339(), "2024-01-02T00:00:00+00:00");
    }
}
