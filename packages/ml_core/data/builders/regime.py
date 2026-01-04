import polars as pl
from pathlib import Path
from .base import AbstractDatasetBuilder


class RegimeDatasetBuilder(AbstractDatasetBuilder):
    """
    Constructs regime datasets.
    automatically switches between 'Structural' and 'Tactical' feature sets
    based on the configured horizon.
    """

    # Critical tickers for Macro/Credit analysis
    MACRO_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG"]

    # Define a buffer to ensure rolling windows (like 63d, 126d) have data
    # 252 days (1 year) is a safe default for macro features.
    WARMUP_BUFFER_DAYS = 252

    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info(
            f"Building Regime Dataset (Horizon: {self.config.target_horizon_days}d)..."
        )

        # 1. Load Macro Data (Pivoted)
        # We need specific tickers to compare asset classes
        macro_df = self._load_macro_data()
        if macro_df.is_empty():
            raise ValueError(
                "Macro data (SPY, TLT, etc.) missing. Cannot build regime model."
            )

        # 2. Load Breadth Data (Aggregated)
        # We calculate this via SQL for maximum speed
        breadth_df = self._load_market_breadth()

        # 3. Join
        df = macro_df.join(breadth_df, on="time", how="left").sort("time")

        # Calculate spy_daily_return for Backtesting
        if "SPY" in df.columns:
            df = df.with_columns(pl.col("SPY").pct_change().alias("spy_daily_return"))

        # 4. Feature Engineering Strategy
        # Branch based on the horizon defined in YAML
        if self.config.target_horizon_days >= 63:
            df = self._build_structural_features(df)
        else:
            df = self._build_tactical_features(df)
            # If we are tactical, we don't want SMA200 nulls to kill our dataset
            cols_to_drop = ["pct_above_sma200", "pct_above_sma50"]
            # Only drop if they exist
            existing_drops = [c for c in cols_to_drop if c in df.columns]
            if existing_drops:
                df = df.drop(existing_drops)

        # 5. Attach Labels (Ground Truth)
        # This loads the GMM clusters we generated earlier
        df = self._attach_labels(df)

        if "__index_level_0__" in df.columns:
            df = df.drop("__index_level_0__")

        # 6. Cleanup
        # Remove nulls created by rolling windows
        # Cast any Decimal columns to Float64
        decimal_cols = [
            c for c, dtype in zip(df.columns, df.dtypes) if dtype == pl.Decimal
        ]
        if decimal_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in decimal_cols])

        return df.drop_nulls()

    def _load_macro_data(self) -> pl.DataFrame:
        """
        Fetches specific macro symbols and pivots them to wide format.
        Output Cols: [time, close_SPY, close_TLT, ...]
        """
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

        # Pivot to Wide Format: One row per date, columns for each symbol
        # This makes vector math (SPY - TLT) incredibly easy
        return raw.pivot(
            index="time", columns="symbol", values="close", aggregate_function="first"
        ).sort("time")

    def _load_market_breadth(self) -> pl.DataFrame:
        """
        Calculates market-wide participation metrics inside the DB.
        """
        # We assume 'sma_50' and 'sma_200' exist in features_daily
        query = f"""
            SELECT 
                fd.time,
                -- Structural
                AVG(CASE WHEN mdd.close > fd.sma_50 THEN 1.0 ELSE 0.0 END)::FLOAT as pct_above_sma50,
                AVG(CASE WHEN mdd.close > fd.sma_200 THEN 1.0 ELSE 0.0 END)::FLOAT as pct_above_sma200,
                -- Tactical
                AVG(CASE WHEN mdd.close > fd.sma_20 THEN 1.0 ELSE 0.0 END)::FLOAT as pct_above_sma20
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            WHERE fd.time >= '{self.config.start_date}'
            GROUP BY fd.time
            ORDER BY fd.time ASC
        """
        return pl.read_database_uri(query, self.db_url)

    def _build_structural_features(self, df: pl.DataFrame) -> pl.DataFrame:
        self.logger.info("Applying STRUCTURAL (Long-Term) Feature Logic...")

        # Helper to get col or null if missing (e.g. if IEF isn't ingested)
        def c(name):
            return pl.col(name) if name in df.columns else pl.lit(None)

        # 1. Trend (Medium-Long)
        # Standard returns
        feats = [
            c("SPY").pct_change(n=63).alias("return_63"),
            c("SPY").pct_change(n=126).alias("return_126"),
        ]

        # 2. Volatility Regime
        # Realized Vol (StdDev of daily returns * sqrt(252))
        # 63d Window
        daily_ret = c("SPY").pct_change()
        feats.append((daily_ret.rolling_std(63) * (252**0.5)).alias("realized_vol_63"))

        # Vol of Vol (Stability of the risk environment)
        # We calculate the StdDev of the Volatility itself
        feats.append((daily_ret.rolling_std(21).rolling_std(63)).alias("vol_of_vol_63"))

        # 3. Macro / Flows (Risk On/Off)
        # SPY (Eq) vs TLT (Rates)
        feats.append(
            (c("SPY").pct_change(63) - c("TLT").pct_change(63)).alias(
                "spy_minus_tlt_63"
            )
        )
        # QQQ (Growth) vs SPY (Market)
        feats.append(
            (c("QQQ").pct_change(63) - c("SPY").pct_change(63)).alias(
                "qqq_minus_spy_63"
            )
        )

        # 4. Credit Stress (The Canary in the Coal Mine)
        # HYG (Junk) vs IEF (Safe Gov Bonds)
        # If Junk outperforms Gov, Risk is On. If Junk crashes relative to Gov, Risk Off.
        feats.append(
            (c("HYG").pct_change(63) - c("IEF").pct_change(63)).alias(
                "credit_spread_proxy_63"
            )
        )

        # 5. Breadth (Pass-through from DB)
        # pct_above_sma50 is already in df

        return df.with_columns(feats)

    def _build_tactical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        self.logger.info("Applying TACTICAL (Short-Term) Feature Logic...")

        def c(name):
            return pl.col(name) if name in df.columns else pl.lit(None)

        feats = []

        # 1. Short Horizon Returns (Shock Detection)
        feats.append(c("SPY").pct_change(5).alias("return_5"))
        feats.append(c("SPY").pct_change(21).alias("return_21"))

        # 2. Volatility & Tail Risk
        # Realized Vol (Short window)
        daily_ret = c("SPY").pct_change()
        feats.append((daily_ret.rolling_std(21) * (252**0.5)).alias("realized_vol_21"))

        # Downside Volatility (Semi-Variance)
        # Filter for only negative returns, then std dev
        # Polars makes this tricky in a rolling window without custom python functions.
        # Approx: Compare rolling Min to rolling Mean.

        # 3. Drawdown (Pain Signal)
        # (Price / RollingMax) - 1
        rolling_max_14 = c("SPY").rolling_max(14)
        feats.append(((c("SPY") / rolling_max_14) - 1).alias("drawdown_14"))

        # 4. Breadth Collapse
        # pct_above_sma20 is in df

        if "SPY" not in df.columns:
            self.logger.error(f"SPY column missing. Available: {df.columns}")
            return df

        return df.with_columns(feats)

    def _attach_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Loads the correct GMM labels based on horizon."""
        horizon = self.config.target_horizon_days
        filename = f"regime_labels_{horizon}d.parquet"
        label_path = (
            Path(__file__).resolve().parents[2] / "labeling" / "artifacts" / filename
        )

        if not label_path.exists():
            # If we are just inferencing/building without training, this might fail.
            # But for a DatasetBuilder used in Training, this MUST exist.
            raise FileNotFoundError(f"Labels {filename} missing.")

        labels = pl.read_parquet(label_path)

        # Inner join to ensure we only have data where we have ground truth
        df = df.join(labels, on="time", how="inner")

        # The parquet has 'regime_label'. The config expects 'target_column'.
        target_col_name = self.config.target_column

        df = df.rename({"regime_label": target_col_name})

        # Ensure target is Int (Classification)
        df = df.with_columns(pl.col(target_col_name).cast(pl.Int32))

        return df
