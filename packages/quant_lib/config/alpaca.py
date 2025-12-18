from pydantic import Field
from pydantic_settings import BaseSettings


class AlpacaConfig(BaseSettings):
    api_key: str = Field(validation_alias="ALPACA_API_KEY")
    secret_key: str = Field(validation_alias="ALPACA_SECRET_KEY")
    paper_trading: bool = Field(validation_alias="ALPACA_PAPER_TRADING", default=False)
