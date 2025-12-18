# packages/database/models.py

from typing import Type
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    BigInteger,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

# This is the base class which our model classes will inherit.
Base = declarative_base()


class Asset(Base):
    __tablename__ = "asset_metadata"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    exchange = Column(String)
    asset_class = Column(String)
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime(timezone=True))


class MarketDataDaily(Base):
    __tablename__ = "market_data_daily"

    time = Column(DateTime(timezone=True), primary_key=True)
    asset_id = Column(Integer, ForeignKey("asset_metadata.id"), primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    vwap = Column(Float)
    trade_count = Column(BigInteger)


# --- Abstract Base for Intraday to reduce code duplication ---
class IntradayBase(Base):
    __abstract__ = True

    time = Column(DateTime(timezone=True), primary_key=True)
    asset_id = Column(Integer, ForeignKey("asset_metadata.id"), primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    vwap = Column(Float)
    trade_count = Column(BigInteger)


# --- Concrete Tables ---
# You can add more here easily (e.g., MarketData1Min, MarketData1Hour)
class MarketData5Min(IntradayBase):
    __tablename__ = "market_data_5min"


# --- Dynamic Dispatcher ---
def get_model_for_timeframe(unit: str, value: int) -> Type[Base]:  # type: ignore
    """
    Returns the SQLAlchemy model class corresponding to the configured timeframe.
    Raises ValueError if the schema doesn't exist.
    """
    key = f"{value}{unit}"  # e.g., "5Minute", "1Hour"

    mapping = {
        "5Minute": MarketData5Min,
        # Add entries here as you create tables
    }

    if key not in mapping:
        raise ValueError(
            f"No database table defined for timeframe '{key}'. "
            f"We strictly support: {list(mapping.keys())}"
        )

    return mapping[key]


class FeaturesDaily(Base):
    __tablename__ = "features_daily"

    # --- Primary Keys ---
    time = Column(DateTime(timezone=True), primary_key=True)
    asset_id = Column(Integer, ForeignKey("asset_metadata.id"), primary_key=True)

    # --- Family: Returns / Momentum ---
    # These are log returns, which are better for ML than percent change
    return_1d = Column(Float)
    return_5d = Column(Float)
    return_21d = Column(Float)  # Approx 1 month
    return_63d = Column(Float)  # Approx 3 months
    return_252d = Column(Float)  # Approx 1 year

    # --- Family: Trend (Moving Averages) ---
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_20 = Column(Float)
    ema_26 = Column(Float)
    ema_50 = Column(Float)

    # --- Family: Oscillators (Momentum Speed) ---
    rsi_14 = Column(Float)

    # MACD (Moving Average Convergence Divergence)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)

    # --- Family: Volatility ---
    atr_14 = Column(Float)  # Average True Range (absolute value)
    atr_14_pct = Column(Float)  # ATR as a percentage of price

    # --- Family: Volume & Money Flow ---
    volume_adv_20 = Column(Float)  # 20-day Average Daily Volume
    relative_volume = Column(Float)  # Today's Volume / ADV_20
    # You can add more here later, like OBV (On-Balance Volume)

    # --- Family: Relative Performance (vs Benchmark) ---
    # rs_spy_normalized = Column(Float)

    __table_args__ = (
        UniqueConstraint("time", "asset_id", name="_daily_features_time_asset_uc"),
    )


# TODO: Add Feature and Prediction tables here later
