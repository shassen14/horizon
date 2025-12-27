# packages/quant_lib/date_utils.py

from datetime import datetime, date, timezone


def get_current_utc_date() -> date:
    """Returns the current date in UTC."""
    return datetime.now(timezone.utc).date()


def ensure_utc_timestamp(dt: datetime | date) -> datetime:
    """Ensures a date or datetime object is a timezone-aware UTC datetime."""
    if isinstance(dt, date) and not isinstance(dt, datetime):
        return datetime(dt.year, dt.month, dt.day, 0, 0, 0, tzinfo=timezone.utc)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)
