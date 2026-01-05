# packages/ml_core/data/builders/regime.py

import polars as pl
from pathlib import Path
from urllib.parse import quote_plus
from .base import AbstractDatasetBuilder


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
        df = macro_df.join(breadth_df, on="time", how="left")

        # Join Technicals
        if not tech_df.is_empty():
            df = df.join(tech_df, on="time", how="left")

        df = df.sort("time")

        # 5. Calculate Helper Columns (if not in DB)
        if "SPY" in df.columns:
            # We calculate this fresh just to be sure we have the exact raw return for backtesting
            df = df.with_columns(pl.col("SPY").pct_change().alias("spy_daily_return"))

        # Calculate Drawdown manually (since it's specific to the window logic)
        # (Price / RollingMax) - 1
        if "SPY" in df.columns:
            rolling_max_14 = pl.col("SPY").rolling_max(14)
            df = df.with_columns(
                ((pl.col("SPY") / rolling_max_14) - 1).alias("drawdown_14")
            )

        # 6. Attach Labels (Ground Truth)
        df = self._attach_labels(df)

        if "__index_level_0__" in df.columns:
            df = df.drop("__index_level_0__")

        return df.drop_nulls()

    def _load_macro_data(self) -> pl.DataFrame:
        """Fetches specific macro symbols and pivots them."""
        symbols_str = ",".join([f"'{s}'" for s in self.MACRO_SYMBOLS])
        query = f"""
            SELECT mdd.time, a.symbol, mdd.close
            FROM market_data_daily mdd
            JOIN asset_metadata a ON mdd.asset_id = a.id
            WHERE a.symbol IN ({symbols_str})
            AND mdd.time >= '{self.config.start_date}'
            ORDER BY mdd.time ASC
        """
        raw = pl.read_database_uri(query, self.db_url)
        if raw.is_empty():
            return pl.DataFrame()

        return raw.pivot(
            index="time", columns="symbol", values="close", aggregate_function="first"
        ).sort("time")

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
        df = df.join(labels, on="time", how="inner")

        # Rename and Cast
        target_col_name = self.config.target_column
        df = df.rename({"regime_label": target_col_name})
        df = df.with_columns(pl.col(target_col_name).cast(pl.Int32))

        return df
