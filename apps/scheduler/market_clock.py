# apps/scheduler/market_clock.py

from datetime import datetime, date, timezone
import pandas as pd
import pandas_market_calendars as mcal


class MarketClock:
    def __init__(self, exchange: str = "NYSE"):
        self.calendar = mcal.get_calendar(exchange)
        self.last_schedule_date = date(1970, 1, 1)
        self.schedule_df = pd.DataFrame()

    def _ensure_schedule(self, target_date: date):
        """Lazy-loads the schedule for the current year to avoid repeated API calls."""
        if target_date.year > self.last_schedule_date.year:
            self.schedule_df = self.calendar.schedule(
                start_date=f"{target_date.year}-01-01",
                end_date=f"{target_date.year}-12-31",
            )
            self.last_schedule_date = target_date

    def is_market_open(self) -> bool:
        """Checks if the market is currently open."""
        now = datetime.now(timezone.utc)
        today = now.date()

        self._ensure_schedule(today)

        try:
            today_schedule = self.schedule_df.loc[today.strftime("%Y-%m-%d")]
            market_open = today_schedule.market_open.to_pydatetime()
            market_close = today_schedule.market_close.to_pydatetime()

            # Ensure they are timezone-aware for comparison
            if market_open.tzinfo is None:
                market_open = market_open.replace(tzinfo=timezone.utc)
            if market_close.tzinfo is None:
                market_close = market_close.replace(tzinfo=timezone.utc)

            return market_open <= now < market_close
        except KeyError:
            # Date not in schedule (weekend/holiday)
            return False
