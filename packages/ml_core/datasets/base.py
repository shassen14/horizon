# packages/ml_core/datasets/base.py

from abc import ABC, abstractmethod
import polars as pl
from packages.ml_core.common.schemas import DataConfig
from packages.quant_lib.config import Settings
from urllib.parse import quote_plus


class AbstractDatasetBuilder(ABC):
    """
    Base class for all dataset strategies.
    Responsible for fetching raw data, calculating specific targets,
    and handling model-specific preprocessing (like regime labeling).
    """

    def __init__(self, settings: Settings, logger, config: DataConfig):
        self.settings = settings
        self.logger = logger
        self.config = config

    @property
    def db_url(self) -> str:
        """Helper to get sync DB URL for Polars."""
        safe_password = quote_plus(self.settings.db.password)
        return (
            f"postgresql://{self.settings.db.user}:{safe_password}@"
            f"{self.settings.db.host}:{self.settings.db.port}/{self.settings.db.name}"
        )

    @abstractmethod
    def load_data(self, start_date: str, end_date: str, **kwargs) -> pl.DataFrame:
        """
        Main entry point. Must return a Polars DataFrame ready for
        temporal processing (lags) and training.
        """
        pass
