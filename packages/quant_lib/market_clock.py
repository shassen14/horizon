# packages/quant_lib/market_clock.py

from datetime import datetime, date, timedelta, timezone
import pandas as pd
import pandas_market_calendars as mcal
from packages.quant_lib.config import settings


class MarketClock:
    """
    Centralized service for Market Hours, Holidays, and Settlement times.
    Shared by Scheduler, Ingestion, and Backtesting.
    """

    def __init__(self, exchange: str = "NYSE"):
        self.calendar = mcal.get_calendar(exchange)
        self.schedule_df = pd.DataFrame()
        self._last_schedule_year = 0

        # We read from settings, but allow overrides if needed
        self.sip_delay = settings.ingestion.sip_delay_minutes

    def _ensure_schedule(self, target_date: date):
        """Lazy-load the calendar schedule for the requested year."""
        if target_date.year != self._last_schedule_year:
            # Load full year to handle boundary conditions easily
            self.schedule_df = self.calendar.schedule(
                start_date=f"{target_date.year}-01-01",
                end_date=f"{target_date.year}-12-31",
            )
            self._last_schedule_year = target_date.year

    def get_session_times(
        self, target_date: date
    ) -> tuple[datetime | None, datetime | None]:
        """
        Returns (market_open_utc, market_close_utc) for a date.
        Returns (None, None) if market is closed.
        """
        self._ensure_schedule(target_date)

        try:
            # Check if date exists in index
            if pd.Timestamp(target_date) not in self.schedule_df.index:
                return None, None

            row = self.schedule_df.loc[target_date.strftime("%Y-%m-%d")]

            # Extract and Force UTC
            m_open = row.market_open.to_pydatetime()
            m_close = row.market_close.to_pydatetime()

            if m_open.tzinfo is None:
                m_open = m_open.replace(tzinfo=timezone.utc)
            else:
                m_open = m_open.astimezone(timezone.utc)

            if m_close.tzinfo is None:
                m_close = m_close.replace(tzinfo=timezone.utc)
            else:
                m_close = m_close.astimezone(timezone.utc)

            return m_open, m_close
        except KeyError:
            return None, None

    def is_market_open(self) -> bool:
        """Is the market open RIGHT NOW?"""
        now = datetime.now(timezone.utc)
        m_open, m_close = self.get_session_times(now.date())

        if not m_open or not m_close:
            return False

        return m_open <= now < m_close

    def is_trading_day(self, target_date: date) -> bool:
        """Is the specific date a trading day?"""
        self._ensure_schedule(target_date)
        return pd.Timestamp(target_date) in self.schedule_df.index

    def is_ingestion_window_open(self) -> bool:
        """
        Is it time to ingest Intraday data?
        Window: Market Open -> Market Close + SIP Delay (15m).
        """
        now = datetime.now(timezone.utc)
        m_open, m_close = self.get_session_times(now.date())

        if not m_open or not m_close:
            return False

        ingest_end = m_close + timedelta(minutes=self.sip_delay)
        return m_open <= now <= ingest_end

    def get_last_settled_date(self) -> date:
        """
        Returns the date of the last COMPLETED trading session.
        - If today is Monday 10 AM -> Returns Friday.
        - If today is Monday 5 PM (Settled) -> Returns Monday.
        - If today is Saturday -> Returns Friday.
        """
        now = datetime.now(timezone.utc)
        today = now.date()

        # m_open, m_close
        _, m_close = self.get_session_times(today)

        # Buffer to ensure data is available (e.g. 45 mins after close)
        settlement_time = timedelta(minutes=45)

        # Logic:
        # 1. If today is a trading day AND we are past the settlement time, today is the answer.
        if m_close and now > (m_close + settlement_time):
            return today

        # 2. Otherwise, walk backwards until we find a trading day.
        # (This handles "During the day", "Before Open", "Weekends", "Holidays")
        candidate = today - timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)

        return candidate

    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Returns the market schedule DataFrame for a range."""
        return self.calendar.schedule(start_date=start_date, end_date=end_date)

    def count_trading_days(
        self, start_date: date, end_date: date, schedule_df: pd.DataFrame = None
    ) -> int:
        """Counts trading days in range using a provided schedule or generating one."""
        if start_date > end_date:
            return 0

        if schedule_df is None:
            schedule_df = self.get_schedule(start_date, end_date)

        # Filter schedule
        mask = (schedule_df.index.date >= start_date) & (
            schedule_df.index.date <= end_date
        )
        return len(schedule_df[mask])
