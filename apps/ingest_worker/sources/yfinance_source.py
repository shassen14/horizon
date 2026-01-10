# apps/ingest_worker/sources/yfinance_source.py
import pandas as pd
import yfinance as yf
import polars as pl
from datetime import datetime
from typing import List, Dict, Any
from packages.quant_lib.interfaces import DataSource


class YFinanceSource(DataSource):
    def __init__(self, logger=None):
        self.logger = logger
        # Maps our internal, clean symbols to what the yfinance API expects.
        self.symbol_map_to_external = {
            "VIX": "^VIX",
            # Add other special symbols here if needed, e.g., "DXY": "DX-Y.NYB"
        }
        # Create the reverse map for cleaning up results
        self.symbol_map_to_internal = {
            v: k for k, v in self.symbol_map_to_external.items()
        }

    def get_all_tickers(self) -> List[Dict[str, Any]]:
        # Yahoo doesn't support "Listing all stocks" easily.
        # We rely on Alpaca for metadata/screening.
        print(
            "⚠️ YFinanceSource does not support 'get_all_tickers'. Use Alpaca for metadata."
        )
        return []

    def get_ohlcv_bars(
        self,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
        timeframe: Any = None,
        use_sip: bool = True,
    ) -> pl.DataFrame:

        if not symbols:
            return pl.DataFrame()

        yf_symbols = [
            self.symbol_map_to_external.get(s.upper(), s.upper()) for s in symbols
        ]

        if self.logger:
            self.logger.info(
                f"YFinanceSource: Requesting {symbols} -> Mapped to {yf_symbols}"
            )

        try:
            start_str = start_dt.strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")

            pdf = yf.download(
                tickers=yf_symbols,
                start=start_str,
                end=end_str,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                prepost=False,
                threads=False,
                progress=False,
            )

            if pdf.empty:
                if self.logger:
                    self.logger.warning(
                        f"yfinance returned empty DataFrame for {yf_symbols}"
                    )
                return pl.DataFrame()

            # --- ROBUST DATA FLATTENING & CLEANING ---

            # Case 1: Multiple Tickers -> MultiIndex
            if isinstance(pdf.columns, pd.MultiIndex):
                # When yfinance uses group_by='ticker', the ticker is level 0 of the columns
                # We stack that level into the index.
                pdf = pdf.stack(level=0, future_stack=True).rename_axis(
                    ["Date", "symbol"]
                )

            # After stacking or for single tickers, reset the index
            pdf.reset_index(inplace=True)

            # --- THE FIX: Use the column cleaner ---
            pdf = _clean_columns(pdf)

            # Rename 'date' to 'time' (now works because columns are simple)
            pdf.rename(columns={"date": "time"}, inplace=True)

            # Ensure 'symbol' column exists
            if "symbol" not in pdf.columns and len(yf_symbols) == 1:
                pdf["symbol"] = yf_symbols[0]

            if "symbol" not in pdf.columns:
                raise ValueError("Could not determine symbol column.")

            # --- Convert to Polars and standardize ---
            df = pl.from_pandas(pdf)

            # Remap symbol back to internal representation ('^VIX' -> 'VIX')
            df = df.with_columns(pl.col("symbol").replace(self.symbol_map_to_internal))

            # Clean and add missing columns
            df = df.with_columns(
                [
                    pl.col("time").dt.replace_time_zone("UTC"),
                    pl.col("volume").fill_null(0).cast(pl.Int64),
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3)
                    .cast(pl.Float64)
                    .alias("vwap"),
                    pl.lit(None).cast(pl.Int64).alias("trade_count"),
                ]
            )

            target_cols = [
                "time",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "trade_count",
            ]
            available = [c for c in target_cols if c in df.columns]

            return df.select(available)

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"YFinance Error for {yf_symbols}: {e}", exc_info=True
                )
            else:
                print(f"YFinance Error: {e}")
            return pl.DataFrame()

    def get_ticker_metadata(self, symbol: str) -> Dict[str, Any] | None:
        """
        Fetches basic metadata for a single symbol from yfinance.
        """
        try:
            ticker = yf.Ticker(symbol)
            # .info is a dictionary containing metadata
            info = ticker.info

            # We only need a few key fields to match the Alpaca structure
            if info:
                return {
                    "symbol": symbol,  # Note: this will be '^VIX'
                    "name": info.get("longName", info.get("shortName")),
                    "exchange": info.get("exchange", "INDEX"),
                    "asset_class": "index",  # We can classify it ourselves
                }
            return None
        except Exception as e:
            # yfinance can be noisy with errors for delisted tickers
            print(f"yfinance metadata error for {symbol}: {e}")
            return None


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles yfinance's inconsistent column naming (strings, tuples, tuple-strings).
    """
    new_cols = []
    for col in df.columns:
        # It might be a tuple ('Close', 'AAPL') or a string "('Close', 'AAPL')"
        # We try to evaluate it if it's a string representation of a tuple
        try:
            # Check if it's a string that looks like a tuple
            if isinstance(col, str) and col.startswith("(") and col.endswith(")"):
                col = eval(col)
        except (SyntaxError, NameError):
            pass  # It's just a regular string

        if isinstance(col, tuple):
            # Take the first element if it's descriptive (e.g., 'Close')
            # Ignore empty strings
            name = col[0] if col[0] else col[1]
            new_cols.append(name.lower().strip())
        else:
            new_cols.append(str(col).lower().strip())

    df.columns = new_cols
    return df
