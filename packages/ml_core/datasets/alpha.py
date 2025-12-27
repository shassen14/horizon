# packages/ml_core/datasets/alpha.py

import polars as pl
import joblib
from pathlib import Path
from .base import AbstractDatasetBuilder


class AlphaDatasetBuilder(AbstractDatasetBuilder):
    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info("Building dataset for Alpha Ranking...")

        start_date = self.config.start_date
        end_date = self.config.end_date

        # 1. Fetch Base Data (All stocks, all features)
        # Note: We perform the join here to get close_price for targets
        query = f"""
            SELECT 
                fd.*, 
                mdd.close as close_price,
                a.symbol
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{start_date}' AND fd.time <= '{end_date}'
            ORDER BY fd.time ASC
        """
        df = pl.read_database_uri(query, self.db_url)
        if df.is_empty():
            return df

        self.logger.info(f"Loaded {len(df):,} rows. Applying Regime Labels...")

        # 2. Apply Regime Labels
        # Only run this if the config specifically asks to filter by regime.
        if self.config.filter_regime is not None:
            self.logger.info("Regime filtering requested. Applying labels...")
            df = self._apply_regime_labels(df)
        else:
            self.logger.info(
                "No regime filter requested. Skipping regime labeling (Baseline Mode)."
            )

        # 3. Calculate Target: 3-Month Forward Return
        # (Standard Ranking Target)
        TARGET_HORIZON_DAYS = 63
        df = df.with_columns(
            (
                (
                    pl.col("close_price").shift(-TARGET_HORIZON_DAYS)
                    / pl.col("close_price")
                )
                - 1
            )
            .over("asset_id")
            .alias("target_forward_return")
        )

        return df

    def _apply_regime_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reconstructs market features and uses the saved Regime Classifier
        to label every day in the dataset as Bull (1) or Bear (0).
        """
        # Determine Model Name from Config
        model_name = self.config.regime_model_name

        # Path to the artifact you just created
        model_path = (
            Path(__file__).resolve().parents[1] / "models" / f"{model_name}.pkl"
        )

        if not model_path.exists():
            self.logger.warning("Regime model not found! Defaulting to '1' (Bull).")
            self.logger.warning(f"model_path: {model_path}")
            return df.with_columns(pl.lit(1).alias("regime"))

        self.logger.info(f"Using regime model: {model_name}")

        try:
            regime_model = joblib.load(model_path)

            # A. Reconstruct Regime Features (Market Breadth & Volatility)
            # We must recreate exactly what the regime model was trained on.
            market_stats = (
                df.group_by("time")
                .agg(
                    [
                        (pl.col("close_price") > pl.col("sma_50"))
                        .sum()
                        .cast(pl.Float64)
                        / pl.count(),  # breadth_sma50_pct
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
            feature_names = regime_model.feature_names_in_
            X_regime = regime_input.select(feature_names).to_pandas()

            regime_preds = regime_model.predict(X_regime)

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
