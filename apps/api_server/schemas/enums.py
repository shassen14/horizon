# apps/api_server/routers/enums.py
from enum import Enum


# System
class SystemHealth(str, Enum):
    HEALTHY = "Healthy"
    DEGRADED = "Degraded"
    STALE = "Stale"
    MAINTENANCE = "Maintenance"


class Environment(str, Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"


# Regime
class RegimeType(str, Enum):
    BULL = "Bull"  # Trending Up
    BEAR = "Bear"  # Trending Down
    SIDEWAYS = "Sideways"  # Choppy/Flat


class RiskLevel(str, Enum):
    RISK_ON = "Risk On"  # Safe to trade aggressively
    RISK_OFF = "Risk Off"  # Cash is king / Protective stops
    NEUTRAL = "Neutral"


# Trade
class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
