# packages/quant_lib/config/features.py

from typing import List
from .base import EnvConfig


class FeatureConfig(EnvConfig):
    # These values should reflect the database table, feature_daily
    # Trend
    # Moving Average Periods
    sma_periods: List[int] = [20, 50, 200]
    ema_periods: List[int] = [12, 20, 26, 50]
    adx_period: int = 14

    # Momentum Indicator Periods
    rsi_period: int = 14  # "Goldilocks" Period
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    # Volatility
    atr_period: int = 14  # standard for measuring recent "choppiness."
    bollinger_period: int = 20
    bollinger_std_dev: int = 2
    # Rate of Change (Momentum) Periods (in trading days)
    roc_periods: List[int] = [1, 5, 21, 63, 126, 252]  # 1d, 1w, 1mo, 3mo, 6mo, 1yr

    # Volume
    mfi_period: int = 14

    # Structure
    structural_periods: List[int] = [252]

    # Stat
    stat_periods: List[int] = [21, 63]

    # factor
    factor_periods: List[int] = [21, 63, 126, 252]
