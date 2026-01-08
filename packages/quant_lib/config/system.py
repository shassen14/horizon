from pathlib import Path
from typing import List
from pydantic import Field
from .base import EnvConfig


class SystemConfig(EnvConfig):
    """
    General system-wide configuration.
    """

    # Maps to HORIZON_ENV in .env
    environment: str = Field(validation_alias="HORIZON_ENV", default="production")

    debug: bool = Field(validation_alias="DEBUG", default=False)
    project_name: str = "Horizon"
    version: str = "1.0.0"

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

    # We use a comma-separated string in .env, but Pydantic can parse it into a list
    # Default includes localhost for development
    allowed_origins: str = Field(
        validation_alias="CORS_ORIGINS",
        default="http://localhost:3000,http://127.0.0.1:3000",
    )

    @property
    def ARTIFACTS_ROOT(self) -> Path:
        return self.PROJECT_ROOT / "artifacts"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [url.strip() for url in self.allowed_origins.split(",")]
