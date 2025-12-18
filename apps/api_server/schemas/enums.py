# apps/api_server/routers/enums.py
from enum import Enum


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class MarketRegimeType(str, Enum):
    RISK_ON = "Risk On"
    RISK_OFF = "Risk Off"
    NEUTRAL = "Neutral"
