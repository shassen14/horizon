# apps/api_server/schemas/history.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class TrendFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    ema_12: float | None = None
    ema_20: float | None = None
    ema_26: float | None = None
    ema_50: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None


class MomentumFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rsi_14: float | None = None
    return_1d: float | None = None
    return_5d: float | None = None
    return_21d: float | None = None  # ~1mo
    return_63d: float | None = None  # ~3mo


class VolatilityFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    atr_14: float | None = None
    atr_14_pct: float | None = None

    bb_upper_20: float | None = None
    bb_middle_20: float | None = None
    bb_lower_20: float | None = None
    bb_width_20: float | None = None
    

class VolumeFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    volume_adv_20: float | None = None
    relative_volume: float | None = None


class FeatureSet(BaseModel):
    """A deeply nested object containing all feature families."""

    trend: TrendFeatures
    momentum: MomentumFeatures
    volatility: VolatilityFeatures
    volume: VolumeFeatures


class HistoryDataPoint(BaseModel):
    """The main, combined data point for the history endpoint."""

    model_config = ConfigDict(from_attributes=True)

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    features: FeatureSet
