import datetime
from zoneinfo import ZoneInfo

import pytest

from horizon.db.types import DailyKey

UTC = datetime.UTC
NEW_YORK = ZoneInfo("America/New_York")


def test_from_trading_date_pins_to_midnight_utc() -> None:
    key = DailyKey.from_trading_date(datetime.date(2024, 1, 2))

    assert key.time.isoformat() == "2024-01-02T00:00:00+00:00"


@pytest.mark.parametrize(
    ("raw", "trading_date"),
    [
        pytest.param(
            datetime.datetime(2024, 7, 1, 13, 30, tzinfo=UTC),
            datetime.date(2024, 7, 1),
            id="summer-market-open",
        ),
        pytest.param(
            datetime.datetime(2024, 1, 3, 2, 0, tzinfo=UTC),
            datetime.date(2024, 1, 2),
            id="winter-utc-date-is-ahead",
        ),
        pytest.param(
            datetime.datetime(2024, 7, 2, 3, 59, tzinfo=UTC),
            datetime.date(2024, 7, 1),
            id="minute-before-new-york-midnight",
        ),
        pytest.param(
            datetime.datetime(2024, 7, 2, 4, 0, tzinfo=UTC),
            datetime.date(2024, 7, 2),
            id="exactly-new-york-midnight",
        ),
    ],
)
def test_from_et_instant_uses_new_york_trading_date(
    raw: datetime.datetime, trading_date: datetime.date
) -> None:
    assert DailyKey.from_et_instant(raw) == DailyKey.from_trading_date(trading_date)


def test_from_et_instant_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        DailyKey.from_et_instant(datetime.datetime(2024, 7, 1, 13, 30))  # noqa: DTZ001


def test_accepts_midnight_utc() -> None:
    value = datetime.datetime(2024, 7, 1, tzinfo=UTC)

    assert DailyKey(value).time == value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(
            datetime.datetime(2024, 7, 1, 9, 30, tzinfo=UTC),
            "midnight",
            id="not-midnight",
        ),
        pytest.param(
            datetime.datetime(2024, 7, 1, 0, 0, 0, 1, tzinfo=UTC),
            "midnight",
            id="one-microsecond-past-midnight",
        ),
        pytest.param(
            datetime.datetime(2024, 7, 1),  # noqa: DTZ001
            "UTC",
            id="naive",
        ),
        pytest.param(
            datetime.datetime(2024, 7, 1, tzinfo=NEW_YORK),
            "UTC",
            id="midnight-in-new-york",
        ),
    ],
)
def test_rejects_invalid_values(value: datetime.datetime, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DailyKey(value)
