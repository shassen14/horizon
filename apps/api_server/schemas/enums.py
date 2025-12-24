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


# Market
class MarketInterval(str, Enum):
    # We map these to Postgres Interval strings
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


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
