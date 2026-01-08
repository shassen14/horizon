# packages/data_pipelines/builders/base.py

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import polars as pl
from urllib.parse import quote_plus
from packages.quant_lib.config import Settings
from packages.contracts.blueprints import DataConfigType


class AbstractDatasetBuilder(ABC):
    def __init__(self, settings: Settings, logger, config: DataConfigType):
        self.settings = settings
        self.logger = logger
        self.config = config
        self.cache_dir = Path(__file__).resolve().parents[3] / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @property
    def db_url(self) -> str:
        safe_password = quote_plus(self.settings.db.password)
        return f"postgresql://{self.settings.db.user}:{safe_password}@{self.settings.db.host}:{self.settings.db.port}/{self.settings.db.name}"

    def get_data(self) -> pl.DataFrame:
        cache_key = self._generate_cache_key()
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if (
            self.config.use_cache
            and not self.config.force_refresh
            and cache_path.exists()
        ):
            self.logger.info(f"⚡ Cache Hit! Loading from {cache_path.name}")
            return pl.read_parquet(cache_path)

        df = self._load_data_internal()

        if self.config.use_cache and not df.is_empty():
            self.logger.info(f"Saving dataset to cache: {cache_path.name}")
            df.write_parquet(cache_path)

        return df

    @abstractmethod
    def _load_data_internal(self) -> pl.DataFrame:
        pass

    def _generate_cache_key(self) -> str:
        if self.config.cache_tag:
            return f"custom_{self.config.cache_tag}"

        config_dict = self.config.model_dump()

        # Exclude feature selection logic from the hash
        config_dict.pop("feature_prefix_groups", None)
        config_dict.pop("feature_exclude_patterns", None)

        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

        return f"dataset_{self.__class__.__name__}_{config_hash}"
