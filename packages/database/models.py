# packages/database/models.py

from typing import Type
from sqlalchemy import (
    Column,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Float,
    Boolean,
    DateTime,
    BigInteger,
    ForeignKey,
    func,
    text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB


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
    # Specific Ledger for 'market_data_daily' data ingestion
    last_market_data_daily_update = Column(DateTime(timezone=True), nullable=True)
    # Specific Ledger for 'market_data_5min' data ingestion
    last_market_data_5min_update = Column(DateTime(timezone=True), nullable=True)


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
    __table_args__ = (
        # 1. Optimizes: "Get history for Symbol X sorted by Time"
        Index("idx_market_data_daily_asset_time", "asset_id", text("time DESC")),
    )


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
    # Note: IntradayBase might define columns, but we define args here on the concrete class
    __table_args__ = (
        # 1. Optimizes: "Get history for Symbol X sorted by Time"
        Index("idx_market_data_5min_asset_time", "asset_id", text("time DESC")),
    )


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
    """
    The Single Source of Truth for asset-specific metrics.
    """

    __tablename__ = "features_daily"

    # --- Identifiers ---
    time = Column(DateTime(timezone=True), nullable=False)
    asset_id = Column(Integer, ForeignKey("asset_metadata.id"), nullable=False)

    # --- 1. Price Trends (Base) ---
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_20 = Column(Float)
    ema_26 = Column(Float)
    ema_50 = Column(Float)

    # --- 2. Momentum & Oscillators ---
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    adx_14 = Column(Float)  # Trend Strength
    mfi_14 = Column(Float)  # Volume-weighted RSI

    # --- 3. Returns (Lookback Windows) ---
    return_1 = Column(Float)
    return_5 = Column(Float)
    return_21 = Column(Float)  # 1 Month
    return_63 = Column(Float)  # 1 Quarter
    return_126 = Column(Float)  # 6 Months
    return_252 = Column(Float)  # 1 Year

    # --- 4. Volatility & Risk ---
    atr_14 = Column(Float)
    atr_14_pct = Column(Float)  # Normalized ATR
    bb_upper_20 = Column(Float)
    bb_middle_20 = Column(Float)
    bb_lower_20 = Column(Float)
    bb_width_20 = Column(Float)  # Volatility Squeeze

    # --- 5. Volume & Liquidity ---
    volume_adv_20 = Column(Float)  # Average Daily Volume
    relative_volume = Column(Float)
    obv = Column(Float)

    # --- 6. Statistical Moments (Distribution Shape) ---
    # Used for Regime detection (Crash risk)
    skew_21 = Column(Float)
    skew_63 = Column(Float)
    zscore_21 = Column(Float)
    zscore_63 = Column(Float)

    # --- 7. Structure (Price Action Context) ---
    high_252_pct = Column(Float)  # Distance from 52-week High
    low_252_pct = Column(Float)  # Distance from 52-week Low

    # --- 8. Factors (Advanced Alpha Signals) ---
    # Relative Strength vs SPY (Benchmark)
    rs_normalized = Column(Float)

    # Volatility-Adjusted Momentum (Sharpe Ratios)
    # Return / Volatility. Higher is better quality momentum.
    sharpe_21 = Column(Float)
    sharpe_63 = Column(Float)
    sharpe_126 = Column(Float)
    sharpe_252 = Column(Float)

    # --- 9. Calendar (Seasonality) ---
    day_of_week = Column(Integer)
    day_of_month = Column(Integer)
    month_of_year = Column(Integer)
    quarter = Column(Integer)

    __table_args__ = (
        PrimaryKeyConstraint("time", "asset_id", name="pk_features_daily"),
        Index("idx_features_daily_asset_time", "asset_id", text("time DESC")),
    )


class MarketContextDaily(Base):
    """
    Stores daily, asset-agnostic indicators describing the overall market environment.
    This table is the single source of truth for the market's "weather".
    """

    __tablename__ = "market_context_daily"

    time = Column(DateTime(timezone=True), primary_key=True)

    # === 1. Volatility / Fear ===
    # Source: VIX data
    vix_close = Column(Float)  # Raw VIX level. High = Fear.
    vix_pct_change_1d = Column(
        Float
    )  # The "shock" factor. A +20% VIX spike is a major event.

    # === 2. Market Internals / Breadth ===
    # Source: Aggregated from all stocks in features_daily
    breadth_pct_above_sma20 = Column(
        Float
    )  # Tactical strength (short-term participation)
    breadth_pct_above_sma50 = Column(Float)  # Mid-term strength (trend health)
    breadth_pct_above_sma200 = Column(
        Float
    )  # Structural strength (long-term bull/bear)

    # === 3. Trend & Momentum (of the market itself) ===
    # Source: SPY features from features_daily
    spy_rsi_14 = Column(Float)  # Is the market overbought/oversold?
    spy_adx_14 = Column(Float)  # Is the market trending strongly (up or down)?

    # === 4. Credit Risk / Financial Stress ===
    # Source: HYG and IEF data from market_data_daily
    # Represents the "High Yield Spread". Widening spread = high stress.
    credit_spread_pct_change_5d = Column(Float)

    # === 5. Interest Rate / Macro Environment ===
    # Source: TLT data from market_data_daily
    # Represents the direction of long-term bond yields. Rising yields = headwind for stocks.
    tlt_pct_change_21d = Column(Float)


class Model(Base):
    """
    Stores metadata about a trained model family.
    Acts as a lightweight, internal model registry.
    """

    __tablename__ = "models"

    model_name = Column(
        String, primary_key=True
    )  # e.g., "alpha_bull", "regime_classifier"
    model_type = Column(String, nullable=False)  # e.g., "ALPHA", "REGIME", "RISK"
    description = Column(String)
    meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())


class Prediction(Base):
    """
    Stores the output of a model for a specific asset at a specific time.
    Designed for high-volume writes and fast "get latest" queries.
    """

    __tablename__ = "predictions"

    # --- Identifiers ---
    time = Column(DateTime(timezone=True), nullable=False)
    asset_id = Column(Integer, ForeignKey("asset_metadata.id"), nullable=False)

    # --- Model Lineage (Human Readable) ---
    model_name = Column(String, ForeignKey("models.model_name"), nullable=False)
    model_version = Column(String, nullable=False)  # e.g., MLflow Run ID or Git Hash

    # --- The Payload ---
    # Flexible JSONB to store any model's output shape
    output = Column(JSONB, nullable=False)

    # --- Configuration ---
    __table_args__ = (
        # Composite Primary Key ensures one prediction per asset/model/version/time
        PrimaryKeyConstraint(
            "time", "asset_id", "model_name", "model_version", name="pk_predictions"
        ),
        # Performance Index for the most common query: "Get latest predictions for a model"
        Index("idx_predictions_model_name_time", "model_name", text("time DESC")),
    )


# class TradeSignal(Base):
#     __tablename__ = "trade_signals"

#     id = Column(Integer, primary_key=True, index=True)
#     generated_at = Column(DateTime(timezone=True), server_default=func.now())

#     # Context
#     asset_id = Column(Integer, ForeignKey("asset_metadata.id"), nullable=False)

#     # Why are we trading? (Traceability)
#     model_name = Column(String)  # e.g. "horizon_alpha"
#     signal_type = Column(String)  # "REBALANCE", "STOP_LOSS", "ENTRY"

#     # The Order
#     direction = Column(String)  # "BUY", "SELL"
#     quantity = Column(Float)  # Number of shares
#     target_weight = Column(Float)  # The desired % of portfolio

#     # The Lifecycle
#     status = Column(
#         String, default="PENDING"
#     )  # PENDING, SENT, FILLED, REJECTED, EXPIRED
#     filled_at = Column(DateTime(timezone=True))
#     filled_price = Column(Float)

#     # Metadata (e.g. limit price, stop price)
#     meta = Column(JSONB)


# class TradeLog(Base):
#     __tablename__ = "trade_log"

#     id = Column(Integer, primary_key=True)
#     timestamp = Column(DateTime(timezone=True), nullable=False)

#     asset_id = Column(Integer, ForeignKey("asset_metadata.id"))

#     # What actually happened
#     action = Column(String)  # "BUY", "SELL"
#     quantity = Column(Float)
#     price = Column(Float)
#     commission = Column(Float, default=0.0)

#     # Link back to the signal (Optional but recommended)
#     signal_id = Column(Integer, ForeignKey("trade_signals.id"), nullable=True)

#     # Broker info (Order ID from Alpaca)
#     broker_order_id = Column(String)


# class PortfolioState(Base):
#     __tablename__ = "portfolio_state"

#     # Composite Key: Snapshot time + Asset
#     # Recommendation: Keep history.

#     time = Column(DateTime(timezone=True), primary_key=True)
#     asset_id = Column(Integer, ForeignKey("asset_metadata.id"), primary_key=True)

#     quantity = Column(Float)
#     cost_basis = Column(Float)
#     current_price = Column(Float)
#     market_value = Column(Float)
#     unrealized_pnl = Column(Float)

#     # Also useful: A separate table for "AccountSummary" (Cash, Buying Power, Total Equity)


# class SystemReport(Base):
#     __tablename__ = "system_reports"

#     id = Column(Integer, primary_key=True, index=True)
#     report_type = Column(String, index=True)  # DATA_QUALITY, MODEL_DRIFT, SYSTEM_HEALTH
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     reference_id = Column(String, nullable=True)  # e.g. "model_v1" or "2024-01-05"
#     content = Column(JSONB, nullable=False)
