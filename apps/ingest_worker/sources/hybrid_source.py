# apps/inges_-worker/sources/hybrid_source.py

import polars as pl
from datetime import datetime
from typing import List, Dict, Any
from alpaca.data.timeframe import TimeFrame

from apps.ingest_worker.sources.yfinance_source import YFinanceSource
from packages.quant_lib.interfaces import DataSource
from .alpaca import AlpacaSource
from .yfinance_source import YFinanceSource

# Define which symbols are "special" and should use the fallback source
YFINANCE_SYMBOLS = ["VIX"]


class HybridSource(DataSource):
    def __init__(self, logger=None):
        self.logger = logger
        self.primary_source = AlpacaSource(logger)
        self.fallback_source = (
            YFinanceSource()
        )  # YFinance source doesn't use logger yet in its __init__

    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Enriched Metadata Fetch:
        1. Gets all assets from the primary source (Alpaca).
        2. Fetches metadata for fallback symbols (VIX) from the secondary source.
        3. Merges the lists, ensuring core symbols are always included.
        """
        if self.logger:
            self.logger.info("HybridSource: Fetching primary asset list from Alpaca...")

        # 1. Primary Fetch
        primary_assets = self.primary_source.get_all_tickers()
        primary_symbols = {asset["symbol"] for asset in primary_assets}

        # 2. Enrichment Fetch
        enriched_assets = []

        for clean_symbol in YFINANCE_SYMBOLS:
            # Check if Alpaca already provided it (unlikely for VIX)
            if clean_symbol in primary_symbols:
                continue

            # Get the yfinance version of the symbol (e.g., '^VIX')
            yf_symbol = self.symbol_map_to_external.get(clean_symbol, clean_symbol)

            if self.logger:
                self.logger.info(
                    f"HybridSource: Enriching metadata for '{clean_symbol}' via yfinance ('{yf_symbol}')..."
                )

            # Fetch metadata
            metadata = self.fallback_source.get_ticker_metadata(yf_symbol)

            if metadata:
                # --- CRITICAL: Un-map the symbol back to our internal 'VIX' ---
                metadata["symbol"] = clean_symbol

                # Add default fields that yfinance doesn't provide
                metadata["tradable"] = metadata.get("tradable", True)  # Assume tradable
                metadata["marginable"] = metadata.get("marginable", False)
                metadata["shortable"] = metadata.get("shortable", False)

                enriched_assets.append(metadata)

        # 3. Merge
        # The final list is all of Alpaca's assets + our special enriched assets
        final_list = primary_assets + enriched_assets

        if self.logger:
            self.logger.success(
                f"HybridSource: Total assets after enrichment: {len(final_list)}"
            )

        return final_list

    def get_ohlcv_bars(
        self,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        use_sip: bool = True,
    ) -> pl.DataFrame:

        # 1. Split Symbols into two groups
        primary_symbols = []
        fallback_symbols = []

        for sym in symbols:
            if sym.upper() in YFINANCE_SYMBOLS:
                # We pass the CLEAN symbol down
                fallback_symbols.append(sym)
            else:
                primary_symbols.append(sym)

        # 2. Fetch from both sources
        df_list = []

        if primary_symbols:
            df_primary = self.primary_source.get_ohlcv_bars(
                primary_symbols, start_dt, end_dt, timeframe, use_sip
            )
            if not df_primary.is_empty():
                df_list.append(df_primary)

        if fallback_symbols:
            df_fallback = self.fallback_source.get_ohlcv_bars(
                fallback_symbols, start_dt, end_dt, timeframe
            )
            if not df_fallback.is_empty():
                df_list.append(df_fallback)

        # 3. Combine results
        if not df_list:
            return pl.DataFrame()

        return pl.concat(df_list)
