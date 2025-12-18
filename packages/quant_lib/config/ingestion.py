from typing import Literal
from pydantic_settings import SettingsConfigDict
from .base import EnvConfig


class IngestionConfig(EnvConfig):
    # API Limits
    max_datapoints: int = 10_000
    max_symbols: int = 200
    safety_ratio: float = 0.90

    # DB Write Limits
    db_chunk_size: int = 1000

    # Caching
    metadata_cache_hours: int = 24

    # Defaults
    default_history_days: int = 5475  # 15 years

    # Intraday
    intraday_timeframe_unit: Literal["Minute", "Hour"] = "Minute"
    intraday_timeframe_value: int = 5
    intraday_lookback_days: int = 1095  # 3 Years

    # Market Microstructure
    market_session_minutes: int = 390
    sip_delay_minutes: int = 15

    # Alpaca Free Tier is 200 requests per minute.
    api_rate_limit_per_minute: int = 200

    # Sleep time between DB writes to let the disk breathe
    db_write_cooldown_seconds: float = 0.5

    model_config = SettingsConfigDict(
        env_prefix="INGESTION_",  # Pydantic will look for INGESTION_MAX_DATAPOINTS, etc.
        case_sensitive=False,
        extra="ignore",
    )
