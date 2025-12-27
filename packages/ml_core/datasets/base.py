# packages/ml_core/datasets/base.py

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
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

        # Define Cache Directory
        self.cache_dir = Path(__file__).resolve().parents[1] / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    @property
    def db_url(self) -> str:
        """Helper to get sync DB URL for Polars."""
        safe_password = quote_plus(self.settings.db.password)
        return (
            f"postgresql://{self.settings.db.user}:{safe_password}@"
            f"{self.settings.db.host}:{self.settings.db.port}/{self.settings.db.name}"
        )

    def get_data(self) -> pl.DataFrame:
        """
        Public entry point. Handles Caching Logic.
        """
        # 1. Generate a unique hash for this configuration
        cache_key = self._generate_cache_key()
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        # 2. Check Cache
        if self.config.use_cache and not self.config.force_refresh:
            if cache_path.exists():
                self.logger.info(f"⚡ Cache Hit! Loading from {cache_path.name}")
                try:
                    return pl.read_parquet(cache_path)
                except Exception as e:
                    self.logger.warning(f"Cache corrupted ({e}). Reloading from DB.")

        # 3. Cache Miss (or Force): Load from DB
        df = self._load_data_internal()

        # 4. Save Cache
        if self.config.use_cache and not df.is_empty():
            self.logger.info(f"Saving dataset to cache: {cache_path.name}")
            df.write_parquet(cache_path)

        return df

    def _generate_cache_key(self) -> str:
        """
        Creates a deterministic hash based on the DataConfig.
        If you change start_date, regime, or builder, the hash changes.
        """
        if self.config.cache_tag:
            return f"custom_{self.config.cache_tag}"

        # Create a dict of config items that affect data definition
        config_dict = {
            "builder": self.config.dataset_builder,
            "start": self.config.start_date,
            "end": self.config.end_date,
            "horizon": self.config.target_horizon_days,
            "regime_model": self.config.regime_model_name,
            # Add prefix groups because changing features means changing data structure
            "features": sorted(self.config.feature_prefix_groups),
        }

        # Serialize to JSON string and Hash
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

        return f"dataset_{self.config.dataset_builder}_{config_hash}"

    @abstractmethod
    def _load_data_internal(self) -> pl.DataFrame:
        """
        Actual implementation to fetch data from DB/Sources.
        (Renamed from load_data)
        """
        pass
