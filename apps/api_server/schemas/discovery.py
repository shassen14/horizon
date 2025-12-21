# apps/api_server/schemas/discovery.py
from pydantic import BaseModel


class ScreenerResult(BaseModel):
    """
    Represents a single asset from the Screener/Market Leaders output.
    """

    rank: int

    # Core Asset Info
    symbol: str
    name: str | None = None

    # Performance Snapshot
    latest_price: float
    daily_change_pct: float

    # Contextual Metrics
    relative_volume: float | None
    rsi_14: float | None
    sma_50_pct_diff: float | None
    atr_14_pct: float | None
