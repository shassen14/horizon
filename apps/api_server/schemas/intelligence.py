# apps/api_server/schemas/intelligence.py
from typing import List
from pydantic import BaseModel
from datetime import datetime
from .enums import RegimeType, RiskLevel


class RankedAsset(BaseModel):
    """
    Represents a single asset from the ML model's ranking output.
    This is a self-contained model.
    """

    rank: int

    # Core Asset Info
    symbol: str
    name: str | None = None

    # Performance Snapshot
    latest_price: float
    daily_change_pct: float

    # Prediction Info
    score: float
    rating: str
    prediction_target: str
    generated_at: datetime
    key_drivers: List[str]


class MarketRegime(BaseModel):
    timestamp: datetime
    regime: RegimeType
    risk_signal: RiskLevel

    # Context scores (0.0 to 1.0)
    trend_score: float  # > 0.5 is Bullish
    volatility_score: float  # > 0.5 is High Volatility

    # The percentage of stocks above their 50-day SMA (0.0 to 1.0)
    # > 0.5 usually confirms a healthy Bull Market.
    # < 0.2 usually signals a washout/bottom.
    breadth_pct: float | None = None

    # The average volatility (ATR %) of the entire market.
    # This is often better than VIX because it measures YOUR universe of stocks.
    market_volatility_avg: float | None = None

    summary: str  # Human readable "The market is in a strong uptrend."
