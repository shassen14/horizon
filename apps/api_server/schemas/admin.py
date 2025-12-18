# apps/api_server/schemas/admin.py
from pydantic import BaseModel
from datetime import datetime
from typing import List
from .enums import SignalAction, SignalStatus


# --- For Portfolio ---
class PositionUpdate(BaseModel):
    """Schema for the POST /sync endpoint body."""

    symbol: str
    quantity: float
    average_buy_price: float


class Position(PositionUpdate):
    """Represents a single position in the portfolio summary."""

    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PortfolioSummary(BaseModel):
    """The main response for the portfolio summary endpoint."""

    total_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    positions: List[Position]


# --- For Signals ---
class TradeSignal(BaseModel):
    """Represents a single trade recommendation from the engine."""

    id: int
    symbol: str
    action: SignalAction
    quantity: float
    reason: str
    status: SignalStatus
    created_at: datetime


class SignalStatusUpdate(BaseModel):
    """Schema for the PUT /signals/{id} endpoint body."""

    status: SignalStatus
    notes: str | None = None
