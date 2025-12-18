# apps/api_server/schemas/assets.py
from pydantic import BaseModel, ConfigDict


class AssetInfo(BaseModel):
    """Lightweight asset information, suitable for search results and lists."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: str | None = None
    exchange: str | None = None


class AssetDetail(AssetInfo):
    """Complete metadata for a single asset."""

    asset_class: str
    is_active: bool
