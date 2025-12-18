# apps/api_server/schemas/intelligence.py
from pydantic import BaseModel
from datetime import datetime
from .enums import MarketRegimeType


class RankedAsset(BaseModel):
    """Represents a single asset in the model's ranking output."""

    rank: int
    symbol: str
    name: str | None = None
    score: float
    latest_price: float
    latest_change_pct: float


class MarketRegime(BaseModel):
    """Represents the current overall market state."""

    regime: MarketRegimeType
    confidence: float
    last_updated: datetime
