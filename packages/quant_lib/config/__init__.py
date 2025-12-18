# packages/quant_lib/config/__init__.py

import sys
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import sub-configs
from .database import DatabaseConfig
from .alpaca import AlpacaConfig
from .ingestion import IngestionConfig
from .screener import ScreenerConfig

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Composition: Grouping configs by domain
    db: DatabaseConfig = DatabaseConfig()
    alpaca: AlpacaConfig = AlpacaConfig()
    ingestion: IngestionConfig = IngestionConfig()
    screener: ScreenerConfig = ScreenerConfig()

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton Instance
try:
    settings = Settings()
except Exception as e:
    print("CRITICAL: Config load failed.")
    raise e
