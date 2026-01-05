# packages/ml_core/training/trainers/regime.py

import polars as pl
import mlflow

from packages.ml_core.common.tracker import ExperimentTracker
from .base import BaseTrainer
from packages.ml_core.modeling.pipeline import HorizonPipeline


class RegimeTrainer(BaseTrainer):
    async def _execute(self, tracker: ExperimentTracker):
        bp = self.blueprint

        # 1. Load Data
        builder = self.factory.create_dataset_builder(bp.data)
        raw_df = builder.get_data()

        if raw_df.is_empty():
            raise ValueError("Dataset empty")

        # Log the dataset using the tracker
        cache_key = builder._generate_cache_key()
        tracker.log_dataset(raw_df, "alpha_raw_data", cache_key)

        # 2. Pipeline (Regime usually has NO Lags)
        # We explicitly pass empty processors list
        pipeline = HorizonPipeline(
            model=self.factory.create_model(bp.model),
            feature_prefixes=bp.data.feature_prefix_groups,
            target=bp.data.target_column,
            processors=[],
            exclude_patterns=bp.data.feature_exclude_patterns,
        )

        # 3. Preprocess
        processed_df = pipeline.preprocess(raw_df)

        self.logger.info(f"DataFrame Columns: {processed_df.columns}")
        self.logger.info(f"Config Prefixes: {bp.data.feature_prefix_groups}")

        # 4. Metadata (Regime needs specific return column for backtest)
        # We look for 'spy_daily_return' which the Builder provided
        meta_cols = ["time"]
        if "spy_daily_return" in processed_df.columns:
            meta_cols.append("spy_daily_return")

        meta_df = processed_df.select(meta_cols).to_pandas()

        self.logger.info(f"Available columns: {processed_df.columns}")
        # 5. Extract X/y
        X, y = pipeline.get_X_y(processed_df)
        pipeline.trained_features = list(X.columns)

        # 6. Split
        split_idx = int(len(X) * bp.training.time_split_ratio)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        meta_val = meta_df.iloc[split_idx:]

        # 7. Train
        strategy = self.factory.create_strategy(bp.training)
        pipeline.model = strategy.train(pipeline.model, X_train, y_train, X_val, y_val)

        # 9. Backtest (Market Timing)
        if bp.backtest.enabled:
            backtester = self.factory.create_backtester(bp.backtest)
            if backtester:
                self.logger.info("Running Regime Market Timing Backtest...")
                preds = pipeline.model.predict(X_val)

                # Check if we have the return column
                if "spy_daily_return" not in meta_val.columns:
                    self.logger.warning("spy_daily_return missing. Skipping backtest.")
                else:
                    bt_df = pl.from_pandas(meta_val).with_columns(
                        [
                            pl.Series(
                                "target_return", meta_val["spy_daily_return"].values
                            ),
                            pl.Series("prediction", preds),
                        ]
                    )

                    bt_metrics = backtester.run(bt_df, "target_return", "prediction")
                    mlflow.log_metrics(
                        {
                            f"bt_{k}": v
                            for k, v in bt_metrics.items()
                            if isinstance(v, (int, float))
                        }
                    )

        return pipeline, X_train, y_train, X_val, y_val
