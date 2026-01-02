import pandas as pd
import yfinance as yf
import polars as pl
from datetime import datetime
from typing import List, Dict, Any
from packages.quant_lib.interfaces import DataSource


class YFinanceSource(DataSource):
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

        # 1. Fetch Data (Vectorized download)
        try:
            start_str = start_dt.strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")

            # Group by ticker ensures (Symbol, Open) structure usually
            pdf = yf.download(
                tickers=symbols,
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
                return pl.DataFrame()

            processed_frames = []

            # 2. Extract and Flatten Data per Symbol
            # Check if we have a MultiIndex (Multiple tickers) or Flat Index (Single ticker)
            is_multi_index = isinstance(pdf.columns, pd.MultiIndex)

            # Helper to process a single symbol's dataframe
            def process_single_symbol_df(sym_df, sym_name):
                # Drop rows where all columns are NaN
                sym_df = sym_df.dropna(how="all")
                if sym_df.empty:
                    return None

                # Reset index to make 'Date' a column
                sym_df = sym_df.reset_index()

                # Flatten columns: Lowercase and strip whitespace
                # yfinance returns 'Date', 'Open', 'High'...
                sym_df.columns = [str(c).lower().strip() for c in sym_df.columns]

                # Rename 'date' to 'time'
                if "date" in sym_df.columns:
                    sym_df = sym_df.rename(columns={"date": "time"})

                # 4. Convert to Polars
                try:
                    pldf = pl.from_pandas(sym_df)
                except Exception:
                    return None

                # A. Handle Volume (Float -> Int)
                if "volume" in pldf.columns:
                    # Fill NaNs with 0, then cast to Int64
                    pldf = pldf.with_columns(
                        pl.col("volume").fill_null(0).cast(pl.Int64)
                    )
                else:
                    pldf = pldf.with_columns(pl.lit(0).cast(pl.Int64).alias("volume"))

                # B. Handle Prices (Ensure Float64)
                for col in ["open", "high", "low", "close"]:
                    if col in pldf.columns:
                        pldf = pldf.with_columns(pl.col(col).cast(pl.Float64))
                    else:
                        # Should not happen with valid data, but safe fallback
                        pldf = pldf.with_columns(
                            pl.lit(0.0).cast(pl.Float64).alias(col)
                        )

                # C. Add Metadata
                pldf = pldf.with_columns(pl.lit(sym_name).alias("symbol"))

                return pldf

            if is_multi_index:
                # Iterate through the top level (Symbols)
                # pdf.columns.levels[0] contains the unique tickers
                for sym in symbols:
                    try:
                        # Extract cross-section for this symbol
                        # This removes the top level (Symbol), leaving just (Open, High...)
                        sym_df = pdf.xs(sym, axis=1, level=0, drop_level=True)
                        pldf = process_single_symbol_df(sym_df, sym)
                        if pldf is not None:
                            processed_frames.append(pldf)
                    except KeyError:
                        continue
            else:
                # Single symbol case (Flat index)
                # The columns are just 'Open', 'High', etc.
                # We assume the user passed 1 symbol or yfinance flattened it.
                # We use the first symbol from the input list.
                sym = symbols[0]
                pldf = process_single_symbol_df(pdf, sym)
                if pldf is not None:
                    processed_frames.append(pldf)

            if not processed_frames:
                return pl.DataFrame()

            # 3. Combine
            df = pl.concat(processed_frames)

            # 4. Standardize Schema
            # Ensure time is UTC
            df = df.with_columns(
                [
                    pl.col("time").dt.replace_time_zone("UTC"),
                    # Since Yahoo doesn't give us VWAP, we calculate (High + Low + Close) / 3
                    # This ensures the column is populated with a mathematically useful value.
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3)
                    .cast(pl.Float64)
                    .alias("vwap"),
                    pl.lit(None).cast(pl.Int64).alias("trade_count"),
                ]
            )

            # Select final schema to match database
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

            # Filter for columns that actually exist (safety check)
            available = [c for c in target_cols if c in df.columns]
            return df.select(available)

        except Exception as e:
            print(f"YFinance Error: {e}")
            return pl.DataFrame()
