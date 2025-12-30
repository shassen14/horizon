# packages/ml_core/training/trainer.py

from datetime import datetime
import getpass
from pathlib import Path
import subprocess
import mlflow
import polars as pl
import pandas as pd

from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.modeling.pipeline import HorizonPipeline, HorizonMLflowWrapper
from packages.ml_core.common.tracker import ExperimentTracker


class Trainer:
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.logger = logger
        self.factory = factory

    async def run(self):
        bp = self.blueprint
        self.logger.info(f"--- Starting Training Pipeline for: {bp.model_name} ---")

        # Start the Context Manager
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # The Tracker handles connection, zombies, and the 'with' block
        with ExperimentTracker(bp.model_name, run_name) as tracker:
            self.logger.info(f"MLflow Run Started. Run ID: {tracker.get_run_id()}")

            # A. Log Context & Config
            # Note: tracker.log_params handles flattening automatically
            self._log_context(bp, tracker)
            tracker.log_params(bp.model_dump(mode="json"))

            # B. Load Data
            try:
                builder = self.factory.create_dataset_builder(bp.data)
                raw_df = builder.get_data()

                # Get the cache key string (e.g. "dataset_Alpha...hash")
                cache_key = builder._generate_cache_key()

                # Log Metadata Only
                tracker.log_dataset(raw_df, name="training_data", cache_key=cache_key)
            except Exception as e:
                self.logger.error(f"Data loading failed: {e}", exc_info=True)
                return

            if raw_df.is_empty():
                self.logger.error("Dataset is empty. Aborting.")
                return

            # C. Build Processor List
            processors = []
            if getattr(bp.data, "generate_lags", False):
                self.logger.info("Adding TemporalFeatureProcessor to pipeline.")
                processors.append(TemporalFeatureProcessor())

            # D. Initialize Pipeline
            try:
                pipeline = HorizonPipeline(
                    model=self.factory.create_model(bp.model),
                    feature_prefixes=bp.data.feature_prefix_groups,
                    target=bp.data.target_column,
                    processors=processors,
                )
            except Exception as e:
                self.logger.error(f"Pipeline init failed: {e}")
                return

            # E. Preprocess
            self.logger.info("Pipeline: Preprocessing data...")
            processed_df = pipeline.preprocess(raw_df)

            # F. Extract X/y
            X, y = pipeline.get_X_y(processed_df)

            if X.empty or y is None or y.empty:
                self.logger.error("Feature extraction failed.")
                return

            # Now that we know which columns matched the prefixes, we save them.
            # This ensures Inference always uses the exact same columns.
            pipeline.trained_features = list(X.columns)

            # Log Shape
            tracker.log_params(
                {
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "feature_names_list": list(X.columns),
                }
            )

            tracker.log_environment()

            # G. Split
            split_idx = int(len(X) * bp.training.time_split_ratio)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            self.logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

            tracker.log_model_signature(X_train.head(5), y_train.head(5))
            profile_df = pl.from_pandas(X_train)
            tracker.log_data_profile(profile_df)

            # H. Train
            strategy = self.factory.create_strategy(bp.training)
            trained_model = strategy.train(
                pipeline.model, X_train, y_train, X_val, y_val
            )
            pipeline.model = trained_model

            # I. Evaluate
            evaluator = self.factory.create_evaluator(bp.training)
            metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
            tracker.log_metrics(metrics)

            # J. Feature Importance
            self._log_feature_importance(pipeline.model, X, tracker)

            # K. Save & Log Artifact
            self.logger.info("Registering model to MLflow...")

            # Wrap the pipeline
            mlflow_model = HorizonMLflowWrapper(pipeline)

            # Infer Signature (Input/Output schema) for the Registry UI
            # We use the validation set X (Pandas) and predictions
            signature = mlflow.models.infer_signature(
                X_val, pipeline.predict(pl.from_pandas(X_val))  # Get sample output
            )

            # We get the list directly from the YAML blueprint
            pip_reqs = bp.model.dependencies

            input_example = X_val.head(5)

            # Log and Register
            # This does 3 things:
            # 1. Pickles the model
            # 2. Uploads it to the NAS (Artifact Store)
            # 3. Creates a version entry in the Postgres DB (Registry)
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=mlflow_model,
                registered_model_name=bp.model_name,  # e.g. "alpha_bull_v1"
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_reqs,
            )

            self.logger.success(
                f"Model registered! Version: {model_info.registered_model_version}"
            )

            self.logger.success(f"Model registered with dependencies: {pip_reqs}")
            self.logger.success(f"Pipeline saved and logged.")

    def _log_context(self, bp, tracker):
        """Logs metadata about the run environment."""
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
            tracker.set_tags({"git_commit": git_commit})
        except Exception:
            tracker.set_tags({"git_commit": "unknown"})

        tracker.set_tags({"user": getpass.getuser(), "model_type": bp.model.type})

    def _log_feature_importance(self, model, X_features, tracker):
        """Logs feature importance if the model supports it."""
        if hasattr(model, "feature_importances_"):
            self.logger.info("Logging Feature Importance...")

            importance_df = pd.DataFrame(
                {
                    "feature": list(X_features.columns),
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)

            imp_path = f"feature_importance.csv"
            importance_df.to_csv(imp_path, index=False)
            tracker.log_artifact(imp_path)
            Path(imp_path).unlink()
