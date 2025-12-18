from typing import List, Optional
from pydantic import computed_field, Field
from pydantic_settings import SettingsConfigDict
from .base import EnvConfig


class ScreenerConfig(EnvConfig):
    asset_classes: str = "us_equity"
    exchanges_allowed: str = "NASDAQ,NYSE,ARCA,BATS"
    exchanges_blocked: str = "OTC"

    min_price: float = Field(default=5.0)
    min_dollar_vol: float = Field(default=10_000_000.0)

    @computed_field
    @property
    def allowed_exchanges_list(self) -> Optional[List[str]]:
        if not self.exchanges_allowed:
            return None
        return [ex.strip().upper() for ex in self.exchanges_allowed.split(",")]

    @computed_field
    @property
    def blocked_exchanges_list(self) -> List[str]:
        if not self.exchanges_blocked:
            return []
        return [ex.strip().upper() for ex in self.exchanges_blocked.split(",")]

    @computed_field
    @property
    def asset_classes_list(self) -> List[str]:
        return [ac.strip().lower() for ac in self.asset_classes.split(",")]

    model_config = SettingsConfigDict(
        env_prefix="SCREENER_",  # Looks for SCREENER_MIN_PRICE, SCREENER_ASSET_CLASSES
        case_sensitive=False,
        extra="ignore",
    )
