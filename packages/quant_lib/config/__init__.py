# packages/quant_lib/config/__init__.py

from pathlib import Path
from pydantic_settings import BaseSettings


# Import sub-configs
from .database import DatabaseConfig
from .alpaca import AlpacaConfig
from .ingestion import IngestionConfig
from .screener import ScreenerConfig
from .features import FeatureConfig
from .system import SystemConfig
from .mlflow import MLflowConfig


# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Composition: Grouping configs by domain
    db: DatabaseConfig = DatabaseConfig()
    mlflow: MLflowConfig = MLflowConfig()
    alpaca: AlpacaConfig = AlpacaConfig()
    ingestion: IngestionConfig = IngestionConfig()
    screener: ScreenerConfig = ScreenerConfig()
    features: FeatureConfig = FeatureConfig()
    system: SystemConfig = SystemConfig()


# Singleton Instance
try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL: Config load failed. Details: {e}")
    raise e
