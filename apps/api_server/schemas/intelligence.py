# apps/api_server/schemas/intelligence.py
from typing import List
from pydantic import BaseModel
from datetime import datetime
from .enums import MarketRegimeType


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
    """Represents the current overall market state."""

    regime: MarketRegimeType
    confidence: float
    last_updated: datetime
