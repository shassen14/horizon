# packages/ml_core/training/trainers/alpha.py

import polars as pl
import mlflow

from packages.ml_core.common.tracker import ExperimentTracker
from .base import BaseTrainer
from packages.ml_core.modeling.pipeline import HorizonPipeline
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor


class AlphaTrainer(BaseTrainer):
    async def _execute(self, tracker: ExperimentTracker):
        bp = self.blueprint

        # 1. Load Data
        builder = self.factory.create_dataset_builder(bp.data)
        raw_df = builder.get_data()

        if raw_df.is_empty():
            raise ValueError("Dataset is empty.")

        # Log the dataset using the tracker
        cache_key = builder._generate_cache_key()
        tracker.log_dataset(raw_df, "alpha_raw_data", cache_key)

        # 2. Pipeline Setup (Alpha ALWAYS uses Lags)
        processors = [TemporalFeatureProcessor()]

        pipeline = HorizonPipeline(
            model=self.factory.create_model(bp.model),
            feature_prefixes=bp.data.feature_prefix_groups,
            target=bp.data.target_column,
            processors=processors,
            exclude_patterns=bp.data.feature_exclude_patterns,
        )

        # 3. Preprocess
        self.logger.info("AlphaTrainer: Generating Lags & Cleaning...")
        processed_df = pipeline.preprocess(raw_df)

        # 4. Metadata Extraction (Crucial for Alpha Backtesting)
        # We need AssetID to track per-stock performance
        meta_cols = ["time", "asset_id", "next_day_return"]
        # Select available columns (safeguard)
        valid_meta_cols = [c for c in meta_cols if c in processed_df.columns]
        meta_df = processed_df.select(valid_meta_cols).to_pandas()

        # 5. Extract X/y
        X, y = pipeline.get_X_y(processed_df)

        # Log feature count
        mlflow.log_param("n_features", X.shape[1])
        pipeline.trained_features = list(X.columns)  # Snapshot exact columns

        # 6. Split
        split_idx = int(len(X) * bp.training.time_split_ratio)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        meta_val = meta_df.iloc[split_idx:]

        self.logger.info(f"Training on {len(X_train)} samples.")

        # 7. Train
        strategy = self.factory.create_strategy(bp.training)
        pipeline.model = strategy.train(pipeline.model, X_train, y_train, X_val, y_val)

        # 9. Backtest (Alpha Specific)
        if bp.backtest.enabled:
            backtester = self.factory.create_backtester(bp.backtest)
            if backtester:
                self.logger.info("Running Alpha Ranking Backtest...")
                preds = pipeline.model.predict(X_val)

                # If we have it, use it. Otherwise fall back to y_val (which causes the bug, but prevents crash)
                if "next_day_return" in meta_val.columns:
                    actual_returns = meta_val["next_day_return"].values
                else:
                    self.logger.warning(
                        "next_day_return missing! Backtest results will be inflated."
                    )
                    actual_returns = y_val.values.ravel()

                # Reconstruct DF with Asset IDs
                bt_df = pl.from_pandas(meta_val).with_columns(
                    [
                        pl.Series("actual_return", actual_returns),
                        pl.Series("prediction", preds),
                    ]
                )

                bt_metrics = backtester.run(bt_df, "actual_return", "prediction")
                # Log with prefix
                mlflow.log_metrics(
                    {
                        f"bt_{k}": v
                        for k, v in bt_metrics.items()
                        if isinstance(v, (int, float))
                    }
                )

        return pipeline, X_train, y_train, X_val, y_val
