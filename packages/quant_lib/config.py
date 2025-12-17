# packages/quant-lib/config.py

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import computed_field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# TODO: Cleanup this file and organize the parameters
# Define the Project Root (3 levels up from this file)
# File: /horizon/packages/quant-lib/config.py -> Root: /horizon
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # --- 1. Database Settings (TimescaleDB) ---
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "horizon_db"

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """
        Constructs the async SQLAlchemy connection string automatically.
        """
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_HOST,
                port=self.POSTGRES_PORT,
                path=self.POSTGRES_DB,
            )
        )

    # --- 2. Alpaca API Settings ---
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_PAPER_TRADING: bool = True

    # --- 3. Data Acquisition & Screener Settings ---
    # Kept from your original logic - very useful for filtering what we fetch
    SCREENER_ASSET_CLASSES: str = "us_equity"
    SCREENER_EXCHANGES_ALLOWED: str = "NASDAQ,NYSE,ARCA,BATS"
    SCREENER_EXCHANGES_BLOCKED: str = "OTC"

    # Quality Thresholds
    # Example: Min $5.00 share price
    SCREENER_MIN_PRICE: float = 5.0
    # Example: Min $2M daily dollar volume (Price * Volume)
    SCREENER_MIN_DOLLAR_VOLUME: float = 2_000_000.0

    # --- Ingestion Constraints ---
    # Max data points (bars) Alpaca returns in one page.
    # Goal: Keep requests under this to avoid pagination (which costs extra requests).
    ALPACA_MAX_DATAPOINTS: int = 10_000

    # Alpaca hard limit on symbols per single request URL
    ALPACA_MAX_SYMBOLS: int = 200

    # Safety buffer: Target 90% of the limit to be safe against data anomalies
    API_SAFETY_RATIO: float = 0.90

    # Database Write Limits (Postgres Parameter Limit safety)
    # 1000 rows * 9 columns = 9000 params (Safe under 65k limit)
    DB_WRITE_CHUNK_SIZE: int = 1000

    # Metadata Cache (skip screener if less than X hours old)
    METADATA_CACHE_HOURS: int = 24

    DEFAULT_HISTORY_DAYS: int = 5475  # 15 years

    # Used to estimate data density for batching
    MARKET_SESSION_MINUTES: int = 390  # Standard NYSE day (6.5 hours)
    SIP_DELAY_MINUTES: int = 15  # Regulation delay for free feeds

    # These control the target resolution and history depth
    INTRADAY_LOOKBACK_DAYS: int = 1095  # 3 years
    INTRADAY_BAR_TIMEFRAME: Literal["Minute", "Hour", "Day"] = "Minute"
    INTRADAY_BAR_TIMEFRAME_VALUE: int = 5  # e.g., 5 for 5-minute bars

    # Computed properties for the screener lists
    @computed_field
    @property
    def ASSET_SCREENER_EXCHANGES_ALLOWED_LIST(self) -> Optional[List[str]]:
        if not self.SCREENER_EXCHANGES_ALLOWED:
            return None
        return [ex.strip().upper() for ex in self.SCREENER_EXCHANGES_ALLOWED.split(",")]

    @computed_field
    @property
    def ASSET_SCREENER_EXCHANGES_BLOCKED_LIST(self) -> List[str]:
        if not self.SCREENER_EXCHANGES_BLOCKED:
            return []
        return [ex.strip().upper() for ex in self.SCREENER_EXCHANGES_BLOCKED.split(",")]

    @computed_field
    @property
    def ASSET_SCREENER_CLASSES_LIST(self) -> List[str]:
        return [ac.strip().lower() for ac in self.SCREENER_ASSET_CLASSES.split(",")]

    # --- 4. Configuration Boilerplate ---
    model_config = SettingsConfigDict(
        # This tells Pydantic where to look for the .env file
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields in .env (like comments or unused vars)
    )


# Instantiate and export
try:
    settings = Settings()
    print(f"INFO: Configuration loaded. Connected to DB at {settings.POSTGRES_HOST}")
except Exception as e:
    print("CRITICAL: Failed to load configuration. Check your .env file.")
    raise e
