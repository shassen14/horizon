# packages/ml_core/datasets/alpha.py

import polars as pl
from .base import AbstractDatasetBuilder
from packages.ml_core.common.registry import ModelRegistryClient


class AlphaDatasetBuilder(AbstractDatasetBuilder):
    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info("Building dataset for Alpha Ranking...")

        # 1. Fetch Base Data (All stocks, all features)
        # Note: We assume 'db_url' property is available from Base class
        query = f"""
            SELECT 
                fd.*, 
                mdd.close as close_price,
                a.symbol
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{self.config.start_date}' AND fd.time <= '{self.config.end_date}'
            ORDER BY fd.time ASC
        """

        try:
            df = pl.read_database_uri(query, self.db_url)
        except Exception as e:
            self.logger.error(f"DB Error: {e}")
            return pl.DataFrame()

        if df.is_empty():
            return df

        # 2. Apply Regime Logic (Specific to Alpha Config)
        # We access specific fields safely because we know self.config is AlphaDataConfig
        if self.config.filter_regime is not None:
            self.logger.info(
                f"Regime Filter Active: Keeping only regime {self.config.filter_regime}"
            )

            # Label the data first
            df = self._apply_regime_labels(df)

            # Filter rows
            if "regime" in df.columns:
                original_count = len(df)
                df = df.filter(pl.col("regime") == self.config.filter_regime)
                self.logger.info(f"Filtered rows from {original_count} to {len(df)}")
            else:
                self.logger.warning(
                    "Regime column missing after labeling. Cannot filter."
                )

        # 3. Calculate Target
        # We look forward N days. We must group by asset_id to avoid shifting data between stocks.
        horizon = self.config.target_horizon_days

        df = df.with_columns(
            ((pl.col("close_price").shift(-horizon) / pl.col("close_price")) - 1)
            .over("asset_id")
            .alias("target_forward_return")
        )

        return df

    def _apply_regime_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reconstructs market features and uses the saved Regime Classifier
        to label every day in the dataset as Bull (1) or Bear (0).
        """
        # Determine Model Name
        model_name = self.config.regime_model_name or "regime_classifier_v1"

        self.logger.info(f"Fetching Regime Model '{model_name}' from Registry...")

        #  Use the Client (Abstracted Access)
        registry = ModelRegistryClient(self.settings.mlflow.tracking_uri)

        # We assume 'production' alias is what we want for training dependencies
        pipeline = registry.load_pipeline(model_name, alias="production")

        if not pipeline:
            self.logger.warning(
                f"Regime model '{model_name}@production' not found. Defaulting to Bull."
            )
            return df.with_columns(pl.lit(1).alias("regime"))

        self.logger.info(f"Using regime model: {model_name}")

        try:
            # A. Reconstruct Regime Features (Market Breadth & Volatility)
            # We must recreate exactly what the regime model was trained on.
            market_stats = (
                df.group_by("time")
                .agg(
                    [
                        (
                            (pl.col("close_price") > pl.col("sma_50"))
                            .sum()
                            .cast(pl.Float64)
                            / pl.count()
                        ).alias("breadth_sma50_pct"),
                        pl.col("atr_14_pct").mean().alias("vol_market_avg_atr_pct"),
                    ]
                )
                .sort("time")
            )

            # B. Get SPY Features
            spy_df = (
                df.filter(pl.col("symbol") == "SPY")
                .select(["time", "rsi_14", "zscore_20"])
                .rename({"rsi_14": "spy_rsi_14", "zscore_20": "spy_zscore_20"})
            )
            # C. Join to create input vector
            regime_input = market_stats.join(spy_df, on="time", how="left").drop_nulls()

            # D. Predict
            # Ensure columns match training order exactly
            regime_preds = pipeline.predict(regime_input)

            # E. Map predictions back to time
            regime_input = regime_input.with_columns(
                pl.Series(name="regime", values=regime_preds)
            )

            # F. Broadcast regime to all stocks
            # Join on 'time' so every stock on 2023-01-01 gets the same regime label
            df_out = df.join(
                regime_input.select(["time", "regime"]), on="time", how="left"
            )

            return df_out

        except Exception as e:
            self.logger.error(f"Failed to apply regime labels: {e}")
            # Fallback
            return df.with_columns(pl.lit(1).alias("regime"))
