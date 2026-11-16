import datetime
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import NewType, Self
from zoneinfo import ZoneInfo

# Plain ints at runtime, distinct types to mypy, so an AssetId can't be passed
# where a RunId is expected.
AssetId = NewType("AssetId", int)

RunId = NewType("RunId", int)


# Where a row's data came from; stored as text in each table's source column.
class Source(StrEnum):
    ALPACA = "alpaca"
    YFINANCE = "yfinance"
    MANUAL = "manual"


# A daily bar's timestamp: the exchange trading date at 00:00 UTC. Sources stamp
# daily bars at different times of day (Alpaca uses midnight New York time), and
# the primary key includes the timestamp, so un-normalized times would store the
# same trading day twice. The database enforces the same rule with a CHECK.
@dataclass(frozen=True)
class DailyKey:
    time: datetime.datetime

    def __post_init__(self) -> None:
        if self.time.utcoffset() != datetime.timedelta(0):
            raise ValueError(f"expected a UTC datetime, got {self.time!r}")

        if self.time.time() != datetime.time.min:
            raise ValueError(f"expected a time of midnight, got {self.time!r}")

    # The trading date is the date in New York, not in UTC: an instant of
    # 2026-11-03T03:00Z is 22:00 on Nov 2 in New York, so it belongs to Nov 2.
    @classmethod
    def from_et_instant(cls, raw: datetime.datetime) -> Self:
        if raw.utcoffset() is None:
            raise ValueError(f"expected a timezone-aware datetime, got {raw!r}")

        return cls.from_trading_date(
            raw.astimezone(ZoneInfo("America/New_York")).date()
        )

    @classmethod
    def from_trading_date(cls, date: datetime.date) -> Self:
        return cls(
            datetime.datetime.combine(date, datetime.time.min, tzinfo=datetime.UTC)
        )


# A bar ready to insert. Prices are raw (unadjusted); splits and dividends are
# stored in corporate_actions and applied when reading.
@dataclass(frozen=True, slots=True)
class NewBar:
    time: DailyKey
    asset_id: AssetId
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    # Optional because not every source provides it. The column is
    # NUMERIC(12, 4), so a 6-decimal value from Alpaca is rounded on insert.
    vwap: Decimal | None
    source: Source


# A row read back from assets.
@dataclass(frozen=True)
class Asset:
    id: AssetId
    symbol: str
    name: str | None
    exchange: str | None
    asset_type: str
    is_active: bool


# An asset before it has a database id; the id is assigned on insert.
@dataclass(frozen=True, slots=True)
class NewAsset:
    symbol: str
    name: str | None
    exchange: str | None
    asset_type: str


@dataclass(frozen=True)
class Split:
    ratio: Decimal


@dataclass(frozen=True)
class Dividend:
    cash: Decimal


# Each kind carries only its own payload. insert_actions maps it onto the table's
# split_ratio / cash_amount columns, leaving the other one NULL.
type ActionKind = Split | Dividend


@dataclass(frozen=True)
class CorporateAction:
    asset_id: AssetId
    ex_date: datetime.date
    source: Source
    action_kind: ActionKind


# How a pipeline run ended. A closed union, so a match over it can end in
# assert_never and mypy flags any case that isn't handled.
@dataclass(frozen=True)
class Success:
    pass


@dataclass(frozen=True)
class Failed:
    message: str  # stored in pipeline_runs.error


@dataclass(frozen=True)
class Skipped:
    pass


type RunOutcome = Success | Failed | Skipped
