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


# TODO: Add Feature and Prediction tables here later
