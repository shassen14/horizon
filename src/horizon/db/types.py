import datetime
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import NewType, Self
from zoneinfo import ZoneInfo

AssetId = NewType("AssetId", int)

RunId = NewType("RunId", int)


class Source(StrEnum):
    ALPACA = "alpaca"
    YFINANCE = "yfinance"
    MANUAL = "manual"


@dataclass(frozen=True)
class DailyKey:
    time: datetime.datetime

    def __post_init__(self) -> None:
        if self.time.utcoffset() != datetime.timedelta(0):
            raise ValueError(f"expected a UTC datetime, got {self.time!r}")

        if self.time.time() != datetime.time.min:
            raise ValueError(f"expected a time of midnight, got {self.time!r}")

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


@dataclass(frozen=True, slots=True)
class NewBar:
    time: DailyKey
    asset_id: AssetId
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None
    source: Source


@dataclass(frozen=True)
class Asset:
    id: AssetId
    symbol: str
    name: str | None
    exchange: str | None
    asset_type: str
    is_active: bool


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


type ActionKind = Split | Dividend


@dataclass(frozen=True)
class CorporateAction:
    asset_id: AssetId
    ex_date: datetime.date
    source: Source
    action_kind: ActionKind


@dataclass(frozen=True)
class Success:
    pass


@dataclass(frozen=True)
class Failed:
    message: str


@dataclass(frozen=True)
class Skipped:
    pass


type RunOutcome = Success | Failed | Skipped
