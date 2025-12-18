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


# Singleton Instance
try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL: Config load failed. Details: {e}")
    raise e
