# packages/data_pipelines/builders/regime.py

import polars as pl
from pathlib import Path
from .base import AbstractDatasetBuilder
from packages.contracts.vocabulary.columns import MarketCol


class RegimeDatasetBuilder(AbstractDatasetBuilder):
    """
    Constructs regime datasets.
    Combines Macro Data (Prices) + Market Breadth (Aggregates) + Technicals (FeaturesDaily).
    """

    MACRO_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG"]

    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info(
            f"Building Regime Dataset (Horizon: {self.config.target_horizon_days}d)..."
        )

        # 1. Load Macro Prices (Pivoted)
        macro_df = self._load_macro_data()
        if macro_df.is_empty():
            raise ValueError("Macro data missing.")

        # 2. Load Market Breadth (SQL Aggregates)
        breadth_df = self._load_market_breadth()

        # 3. Load Technical Features (From features_daily for SPY)
        tech_df = self._load_spy_technicals()

        # 4. Join All Sources
        # We start with macro_df as the base
        df = macro_df.join(breadth_df, on=MarketCol.TIME, how="left")

        # Join Technicals
        if not tech_df.is_empty():
            df = df.join(tech_df, on=MarketCol.TIME, how="left")

        # 5. Attach Labels (Ground Truth)
        df = self._attach_labels(df)

        if "__index_level_0__" in df.columns:
            df = df.drop("__index_level_0__")

        return df.drop_nulls()

    def _load_macro_data(self) -> pl.DataFrame:
        """
        Fetches SPY as full OHLC (for Permutation) and other macros as Close-only (for Features).
        """
        # 1. Separate SPY from the rest
        anchor_symbol = "SPY"
        other_symbols = [s for s in self.MACRO_SYMBOLS if s != anchor_symbol]

        # 2. Fetch SPY (Full OHLCV)
        # We need this to exist for the OHLCPermutator to work
        query_spy = f"""
            SELECT 
                mdd.time, 
                mdd.open, 
                mdd.high, 
                mdd.low, 
                mdd.close, 
                mdd.volume,
                -- We alias close to 'SPY' so the Feature Logic (spy_minus_tlt) works later
                mdd.close as "{anchor_symbol}" 
            FROM market_data_daily mdd
            JOIN asset_metadata a ON mdd.asset_id = a.id
            WHERE a.symbol = '{anchor_symbol}'
            AND mdd.time >= '{self.config.start_date}'
            ORDER BY mdd.time ASC
        """
        spy_df = pl.read_database_uri(query_spy, self.db_url)

        # 3. Fetch Others (Close Only -> Pivoted)
        if other_symbols:
            others_str = ",".join([f"'{s}'" for s in other_symbols])
            query_others = f"""
                SELECT mdd.time, a.symbol, mdd.close
                FROM market_data_daily mdd
                JOIN asset_metadata a ON mdd.asset_id = a.id
                WHERE a.symbol IN ({others_str})
                AND mdd.time >= '{self.config.start_date}'
                ORDER BY mdd.time ASC
            """
            others_raw = pl.read_database_uri(query_others, self.db_url)

            if not others_raw.is_empty():
                # Pivot: Time | TLT | HYG | IEF ...
                others_pivoted = others_raw.pivot(
                    index="time",
                    columns="symbol",
                    values="close",
                    aggregate_function="first",
                ).sort("time")

                # 4. Join SPY + Others
                # Left join ensures we stick to the SPY timeline
                df = spy_df.join(others_pivoted, on="time", how="left")
            else:
                df = spy_df
        else:
            df = spy_df

        # 5. Type Safety (Polars sometimes implies Decimal for prices)
        # Cast everything except Time to Float64
        float_cols = [c for c, t in zip(df.columns, df.dtypes) if c != "time"]
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])

        return df.sort("time")

    def _load_market_breadth(self) -> pl.DataFrame:
        """Calculates market-wide participation metrics."""
        query = f"""
            SELECT 
                fd.time,
                AVG(CASE WHEN mdd.close > fd.sma_50 THEN 1.0 ELSE 0.0 END)::FLOAT as pct_above_sma50,
                AVG(CASE WHEN mdd.close > fd.sma_200 THEN 1.0 ELSE 0.0 END)::FLOAT as pct_above_sma200,
                AVG(CASE WHEN mdd.close > fd.sma_20 THEN 1.0 ELSE 0.0 END)::FLOAT as pct_above_sma20
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            WHERE fd.time >= '{self.config.start_date}'
            GROUP BY fd.time
            ORDER BY fd.time ASC
        """
        return pl.read_database_uri(query, self.db_url)

    def _load_spy_technicals(self) -> pl.DataFrame:
        """
        Fetches pre-calculated statistics from features_daily for SPY.
        """
        query = f"""
            SELECT 
                fd.time,
                -- Return Families
                fd.return_5, fd.return_21, fd.return_63,
                -- Volatility Families
                fd.atr_14_pct as realized_vol_21, -- Mapping ATR to generic vol name if desired
                fd.bb_width_20,
                -- Trend Families
                fd.adx_14,
                fd.rsi_14,
                -- Stat Families
                fd.skew_20, fd.skew_60,
                fd.zscore_20, fd.zscore_60
            FROM features_daily fd
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE a.symbol = 'SPY'
            AND fd.time >= '{self.config.start_date}'
            ORDER BY fd.time ASC
        """
        return pl.read_database_uri(query, self.db_url)

    def _attach_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Loads the correct GMM labels based on horizon."""
        horizon = self.config.target_horizon_days
        filename = f"regime_labels_{horizon}d.parquet"
        label_path = (
            Path(__file__).resolve().parents[2] / "labeling" / "artifacts" / filename
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Labels {filename} missing. Run labeling engine first."
            )

        labels = pl.read_parquet(label_path)

        # Inner join to ensure we only have data where we have ground truth
        return df.join(labels, on=MarketCol.TIME, how="inner")
