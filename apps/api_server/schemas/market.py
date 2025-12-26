# apps/api_server/schemas/history.py
from typing import List
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
    adx_14: float | None = None


class MomentumFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rsi_14: float | None = None
    return_1: float | None = None
    return_5: float | None = None
    return_21: float | None = None
    return_63: float | None = None
    return_126: float | None = None
    return_252: float | None = None


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
    obv: float | None = None
    mfi_14: float | None = None


class StructuralFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    high_252_pct: float | None = None
    low_252_pct: float | None = None


class StatFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    skew_20: float | None = None
    skew_60: float | None = None
    zscore_20: float | None = None
    zscore_60: float | None = None


class CalendarFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    day_of_week: int | None = None
    day_of_month: int | None = None
    month_of_year: int | None = None
    quarter: int | None = None


class FeatureSet(BaseModel):
    """A deeply nested object containing all feature families."""

    trend: TrendFeatures
    momentum: MomentumFeatures
    volatility: VolatilityFeatures
    volume: VolumeFeatures
    structure: StructuralFeatures
    stats: StatFeatures
    calendar: CalendarFeatures


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


class MarketSnapshot(BaseModel):
    symbol: str
    price: float
    change_1d: float | None = None
    change_1d_pct: float | None = None
    last_updated: datetime
    sparkline: List[float]  # Simple list of recent closing prices
