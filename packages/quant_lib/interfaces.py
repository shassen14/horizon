# packages/quant_lib/interfaces.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Any, Dict
import polars as pl


class DataSource(ABC):
    """
    Abstract Base Class for all Financial Data Sources.
    Any new provider (Polygon, Yahoo, etc.) must inherit from this.
    """

    @abstractmethod
    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Must return a list of dictionaries representing assets.
        Required keys in dict: 'symbol', 'exchange', 'asset_class', 'name'.
        """
        pass

    @abstractmethod
    def get_ohlcv_bars(
        self,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
        timeframe: Any,  # Kept generic (Any) to allow source-specific objects
        use_sip: bool = True,
    ) -> pl.DataFrame:
        """
        Must return a Polars DataFrame with columns:
        [time, open, high, low, close, volume, trade_count, vwap, symbol]
        """
        pass
