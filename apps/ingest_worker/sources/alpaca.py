# apps/ingest-worker/sources/alpaca.py

import polars as pl
from datetime import datetime
from typing import List, Optional, Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus

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
    def __init__(self, logger=None):
        self.logger = logger

        self.api_key = settings.alpaca.api_key
        self.secret_key = settings.alpaca.secret_key

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key must be set in .env file.")

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(
            self.api_key, self.secret_key, paper=settings.alpaca.paper_trading
        )

    def get_all_tickers(self) -> List[dict]:
        """Fetches all active, tradable US Equity assets from Alpaca."""
        if self.logger:
            self.logger.info("Fetching all active assets from Alpaca...")

        request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class="us_equity")
        assets = self.trading_client.get_all_assets(request)

        return [
            {
                "symbol": asset.symbol,
                "name": asset.name,
                "exchange": asset.exchange.value if asset.exchange else "UNKNOWN",
                "asset_class": "us_equity",
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
            }
            for asset in assets
            if asset.tradable
        ]

    def get_ohlcv_bars(
        self,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        use_sip: bool = True,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()

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
            # Debug log to trace requests
            if self.logger:
                self.logger.debug(
                    f"Requesting {len(symbols)} syms | {start_dt} -> {end_dt} | TF: {timeframe}"
                )

            barset = self.client.get_stock_bars(request_params)

            data = []
            for symbol in barset.data:
                for bar in barset.data[symbol]:
                    bar_data = bar.model_dump()
                    bar_data["symbol"] = symbol
                    data.append(bar_data)

            if not data:
                return pl.DataFrame()

            df = pl.from_dicts(data)
            df = df.rename(ALPACA_COLUMN_MAP)

            df = df.with_columns(
                [
                    pl.col("time").dt.replace_time_zone("UTC"),
                    pl.col("volume").cast(pl.Int64),
                    pl.col("trade_count").cast(pl.Int64),
                ]
            )

            return df

        except Exception as e:
            msg = f"Error fetching data from Alpaca: {e}"
            context = (
                f"Params: Start={start_dt}, End={end_dt}, Feed={feed}, TF={timeframe}"
            )

            if self.logger:
                self.logger.error(msg)
                self.logger.error(context)
            else:
                # Fallback if no logger injected
                print(msg)
                print(context)

            return pl.DataFrame()
