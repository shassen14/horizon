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
