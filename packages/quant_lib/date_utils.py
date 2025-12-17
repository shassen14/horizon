# packages/quant_lib/date_utils.py

from datetime import datetime, date, timedelta, timezone
import pandas_market_calendars as mcal


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


def get_market_close_yesterday() -> datetime:
    """
    Returns a UTC datetime representing the end of yesterday (or today if after close).
    For safety with Alpaca Free Tier, we default to yesterday's date.
    """
    today = get_current_utc_date()
    yesterday = today - timedelta(days=1)
    # Return as datetime at end of day, or just start of day depending on API needs
    # Alpaca 'end' is inclusive, so start of today works as "up until today"
    return datetime(
        yesterday.year, yesterday.month, yesterday.day, 23, 59, 59, tzinfo=timezone.utc
    )


def get_full_trading_schedule(start_date: date, end_date: date, exchange: str = "NYSE"):
    nyse = mcal.get_calendar(exchange)
    return nyse.schedule(start_date=start_date, end_date=end_date)


def get_trading_days_in_range(start_date: date, end_date: date, schedule_df=None):
    if start_date > end_date:
        return 0
    if schedule_df is None:
        schedule_df = get_full_trading_schedule(start_date, end_date)

    # Filter the schedule for our range
    # Ensure strict date comparison
    mask = (schedule_df.index.date >= start_date) & (schedule_df.index.date <= end_date)
    return len(schedule_df[mask])
