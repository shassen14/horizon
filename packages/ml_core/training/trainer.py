# packages/ml_core/training/trainer.py

from pathlib import Path
import polars as pl

from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.factory import MLComponentFactory
from packages.ml_core.datasets.processors import TemporalFeatureProcessor
import joblib


class Trainer:
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.logger = logger
        self.factory = factory
        self.temporal_feature_processor = TemporalFeatureProcessor()

    async def run(self):
        """Executes the full training pipeline."""
        bp = self.blueprint
        self.logger.info("--- Starting Model Training Pipeline ---")

        # 1. Load Data (Via Factory)
        builder = self.factory.create_dataset_builder(bp.data)
        base_df = builder.get_data()

        if base_df.is_empty():
            self.logger.error("Data loading failed. Aborting training.")
            return

        if bp.data.filter_regime is not None:
            if "regime" not in base_df.columns:
                self.logger.warning("Filter requested but 'regime' column missing.")
            else:
                original_count = len(base_df)
                base_df = base_df.filter(pl.col("regime") == bp.data.filter_regime)
                filtered_count = len(base_df)
                self.logger.info(
                    f"Regime Filter ({bp.data.filter_regime}): Reduced data from {original_count} to {filtered_count} rows."
                )

        if base_df.is_empty():
            self.logger.error("No data left after regime filtering.")
            return

        # 2 ML Feature Engineering (Conditional)
        if bp.data.generate_lags:
            self.logger.info("Generating temporal features (lags/deltas)...")
            try:
                model_df = self.temporal_feature_processor.prepare_features(base_df)
            except Exception as e:
                self.logger.error(f"Feature processing failed: {e}")
                return
        else:
            self.logger.info("Skipping temporal features (Regime Model detected).")
            model_df = base_df

        # 3. Clean final dataset
        model_df = model_df.drop_nulls()

        # 4. Select Feature Columns (Dynamically)
        all_cols = model_df.columns
        # Filter columns that match the prefixes defined in YAML
        feature_cols = [
            c
            for c in all_cols
            if any(c.startswith(p) for p in bp.data.feature_prefix_groups)
        ]

        if not feature_cols:
            self.logger.error(
                f"No feature columns found matching prefixes: {bp.data.feature_prefix_groups}"
            )
            self.logger.info(f"Available columns were: {all_cols}")
            return

        # Determine target column
        if bp.model.objective == "classification":
            target_col = "target_regime_bull"
        else:
            target_col = "target_forward_return"

        if target_col not in model_df.columns:
            self.logger.error(f"Target column '{target_col}' not found in dataset.")
            return

        self.logger.info(
            f"Training with {len(feature_cols)} features. Target: {target_col}"
        )

        # Convert to Pandas for Scikit-Learn / LightGBM
        X = model_df.select(feature_cols).to_pandas()
        y = model_df.select(target_col).to_pandas()

        # 5. Time-Series Split
        split_idx = int(len(X) * bp.training.time_split_ratio)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.logger.info(
            f"Training on {len(X_train)} samples, Validating on {len(X_val)}..."
        )

        # 6. Get Components from Factory
        try:
            model = self.factory.create_model(bp.model)
            strategy = self.factory.create_strategy(bp.training)
            evaluator = self.factory.create_evaluator(bp.training)
        except Exception as e:
            self.logger.error(f"Factory configuration error: {e}")
            return

        # 7. Model Training
        trained_model = strategy.train(model, X_train, y_train, X_val, y_val)

        # 8. Run Evaluation (Generic Call)
        self.logger.info("--- Evaluating Model Performance ---")
        metrics = evaluator.evaluate(trained_model, X_val, y_val, self.logger)
        self.logger.info(f"Final Metrics: {metrics}")

        # 9. Save Artifact
        # Note: PyTorch models should save state_dict, not the whole model, but joblib works for simple cases
        model_dir = Path(__file__).resolve().parents[1] / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{bp.model_name}.pkl"

        joblib.dump(trained_model, model_path)
        self.logger.success(f"Model saved to {model_path}")
