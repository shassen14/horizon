import mlflow
import pandas as pd
import polars as pl
import getpass
import subprocess
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.modeling.pipeline import HorizonMLflowWrapper
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.common.tracker import ExperimentTracker


class BaseTrainer(ABC):
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.factory = factory
        self.logger = logger
        self.settings = factory.settings

    async def run(self):
        """
        Template method that handles the full MLOps lifecycle using ExperimentTracker.
        """
        bp = self.blueprint
        self.logger.info(f"--- Starting Pipeline: {bp.model_name} ---")

        # The tracker's __enter__ handles all connection, setup, and teardown.
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"

        with ExperimentTracker(bp.model_name, run_name) as tracker:
            self.logger.info(f"MLflow Run Started. ID: {tracker.get_run_id()}")

            try:
                # 1. Log Run Context (Git, User, Config)
                self._log_context(tracker)

                # 2. Execute Subclass Logic to get trained pipeline and data splits
                pipeline, X_train, y_train, X_val, y_val = await self._execute(tracker)

                # 3. Log all post-training artifacts
                if pipeline:
                    self._log_artifacts(
                        tracker, pipeline, X_train, y_train, X_val, y_val
                    )

            except Exception as e:
                self.logger.exception("Training pipeline crashed")
                # The tracker's __exit__ method will mark the run as failed.
                raise e

    @abstractmethod
    async def _execute(self, tracker: ExperimentTracker) -> tuple:
        """
        Subclasses must implement this.
        Must return: (trained_pipeline, X_train, y_train, X_val, y_val)
        """
        pass

    def _log_context(self, tracker: ExperimentTracker):
        """Logs metadata about the run environment and configuration."""
        bp = self.blueprint

        # Log Git hash and user
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

        # Log flattened config params
        tracker.log_params(bp.model_dump(mode="json"))

        # Log minimal environment
        tracker.log_environment(bp.model.dependencies)

    def _log_artifacts(
        self, tracker: ExperimentTracker, pipeline, X_train, y_train, X_val, y_val
    ):
        """Logs all post-training artifacts: metrics, importance, profile, and the model itself."""

        # 1. Log ML Metrics
        self.logger.info("Logging ML metrics...")
        evaluator = self.factory.create_evaluator(self.blueprint.training)
        metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
        tracker.log_metrics(metrics)

        # 2. Log Data Profile
        self.logger.info("Logging data profile...")
        tracker.log_data_profile(pl.from_pandas(X_train))

        # 3. Log Feature Importance
        self.logger.info("Logging feature importance...")
        self._log_feature_importance(pipeline.model, X_train, tracker)

        # 4. Save, Register, and Log Model Signature
        self._save_and_register(pipeline, X_val, y_val, tracker)

    def _log_feature_importance(
        self, model, X_features: pd.DataFrame, tracker: ExperimentTracker
    ):
        """Logs feature importance as a CSV artifact."""
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame(
                {
                    "feature": list(X_features.columns),
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)

            imp_path = "feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            tracker.log_artifact(imp_path)
            Path(imp_path).unlink()

    def _save_and_register(
        self,
        pipeline,
        X_sample: pd.DataFrame,
        y_sample: pd.DataFrame,
        tracker: ExperimentTracker,
    ):
        """Saves, registers, and logs the model with its signature."""
        bp = self.blueprint
        self.logger.info("Registering model to MLflow...")

        # Prepare Signature and Example
        input_example = X_sample.head(5)

        try:
            prediction_sample = pipeline.predict(pl.from_pandas(input_example))
        except Exception:
            prediction_sample = y_sample.head(5)

        # Infer signature
        signature = tracker.log_model_signature(input_example, prediction_sample)

        # Wrap and Log Model
        mlflow_model = HorizonMLflowWrapper(pipeline)

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=mlflow_model,
            registered_model_name=bp.model_name,
            pip_requirements=bp.model.dependencies,
            input_example=input_example,
            signature=signature,
        )

        self.logger.success(
            f"Model registered! Version: {model_info.registered_model_version}"
        )
