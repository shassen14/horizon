# packages/quant_lib/config/base.py

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define Project Root (4 levels up from this file)
# File: /horizon/packages/quant_lib/config/base.py -> Root: /horizon
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EnvConfig(BaseSettings):
    """
    A base config that all other configs inherit from.
    Its only job is to specify the location of the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
