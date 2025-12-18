# apps/ingest-worker/sources/alpaca.py

import polars as pl
from datetime import datetime
from typing import Any, Dict, List, Union
from alpaca.common.types import RawData
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.models.bars import BarSet
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass

from packages.quant_lib.config import settings
from packages.quant_lib.interfaces import DataSource


# A mapping to rename Alpaca's columns to ours
ALPACA_COLUMN_MAP = {
    "timestamp": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "trade_count": "trade_count",
    "vwap": "vwap",
}


class AlpacaSource(DataSource):
    def __init__(self):
        self.api_key = settings.alpaca.api_key
        self.secret_key = settings.alpaca.secret_key
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key must be set in .env file.")

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(
            self.api_key, self.secret_key, paper=settings.alpaca.paper_trading
        )

    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """Fetches all active, tradable US Equity assets from Alpaca."""
        request = GetAssetsRequest(
            status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY
        )
        assets = self.trading_client.get_all_assets(request)

        # Convert to list of dicts
        return [
            {
                "symbol": asset.symbol,  # type: ignore
                "name": asset.name,  # type: ignore
                "exchange": asset.exchange.value if asset.exchange else "UNKNOWN",  # type: ignore
                "asset_class": "us_equity",
                "tradable": asset.tradable,  # type: ignore
                "marginable": asset.marginable,  # type: ignore
                "shortable": asset.shortable,  # type: ignore
            }
            for asset in assets
            if asset.tradable  # type: ignore
        ]

    def get_ohlcv_bars(
        self,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
        timeframe: TimeFrame = TimeFrame.Day,  # Default to Daily
        use_sip: bool = True,
    ) -> pl.DataFrame:
        """
        Fetches OHLCV data.
        :param timeframe: Alpaca TimeFrame object (e.g., TimeFrame.Day, TimeFrame(5, TimeFrameUnit.Minute))
        :param use_sip: If True, uses SIP feed (Complete data). If False, uses IEX (Free/Real-time but sparse).
        """
        if not symbols:
            return pl.DataFrame()

        # Handle Data Feed Selection
        # SIP is high quality but delayed 15m on free plans.
        # IEX is real-time on free plans but misses volume/trades from other exchanges.
        feed = DataFeed.SIP if use_sip else DataFeed.IEX

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            adjustment="all",
            feed=feed,
        )

        try:
            barset: Union[BarSet, RawData] = self.client.get_stock_bars(request_params)

            # Convert the Alpaca BarSet object to a list of dicts
            data = []
            if isinstance(barset, BarSet):
                for symbol in barset.data:
                    for bar in barset.data[symbol]:
                        bar_data = bar.model_dump()
                        bar_data["symbol"] = symbol  # Add symbol to each bar's dict
                        data.append(bar_data)
            else:
                print("Expected barset to be of type, BarSet")

            if not data:
                return pl.DataFrame()

            # Create Polars DataFrame and perform transformations
            df = pl.from_dicts(data)
            df = df.rename(ALPACA_COLUMN_MAP)

            # Ensure correct data types
            df = df.with_columns(
                [
                    pl.col("time").dt.replace_time_zone("UTC"),
                    pl.col("volume").cast(pl.Int64),
                    pl.col("trade_count").cast(pl.Int64),
                ]
            )

            return df

        except Exception as e:
            print(f"Error fetching data from Alpaca: {e}")
            return pl.DataFrame()

    # We will add get_all_tradable_assets here later
