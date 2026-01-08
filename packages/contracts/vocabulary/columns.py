from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class MarketCol(StrEnum):
    """Standard columns from raw market data providers."""

    TIME = "time"
    SYMBOL = "symbol"
    ASSET_ID = "asset_id"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"


class FeatureCol(StrEnum):
    """Standard names for commonly calculated features."""

    # Trend
    SMA_20 = "sma_20"
    SMA_50 = "sma_50"
    SMA_200 = "sma_200"

    # Momentum
    RSI_14 = "rsi_14"

    # Volatility
    ATR_14_PCT = "atr_14_pct"
    BB_WIDTH_20 = "bb_width_20"

    # Breadth
    PCT_ABOVE_SMA50 = "pct_above_sma50"


class RegimeCol(StrEnum):
    """Columns specific to the Regime Classification domain."""

    FWD_RET = "regime_fwd_ret"  # Prefixed to avoid collision with Alpha targets
    FWD_VOL = "regime_fwd_vol"
    TARGET = "target_regime"  # The final 0, 1, 2... label


class AlphaCol(StrEnum):
    """Columns specific to the Alpha Ranking domain."""

    TARGET_RETURN = "target_return_fwd"  # e.g., 63-day forward return for an asset
    SCORE = "alpha_score"  # The final blended 0-1 score
    RANK = "alpha_rank"  # The ordinal rank (1st, 2nd, 3rd...)


class RiskCol(StrEnum):
    """Columns specific to the Risk Forecasting domain."""

    TARGET_VOL = "target_volatility"  # e.g., 21-day forward realized volatility
    TARGET_ES_95 = "target_expected_shortfall_95"  # 95% Expected Shortfall
