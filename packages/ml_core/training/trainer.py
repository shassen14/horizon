# packages/ml_core/training/trainer.py

from datetime import datetime
import getpass
from pathlib import Path
import subprocess
import mlflow
import pandas as pd

from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.modeling.pipeline import HorizonPipeline
from packages.quant_lib.config import settings


class Trainer:
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.logger = logger
        self.factory = factory

    async def run(self):
        bp = self.blueprint
        self.logger.info(f"--- Starting Training Pipeline for: {bp.model_name} ---")

        # 1. Configuration (Safe operations)
        try:
            mlflow.set_tracking_uri(settings.system.mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=bp.model_name)
        except mlflow.MlflowException as e:
            self.logger.error(f"MLflow connection failed: {e}")
            return

        # 2. Zombie Killer
        # Ensure absolutely NO runs are active before we start our own.
        # We loop because sometimes nested runs can leave multiple layers open.
        while mlflow.active_run():
            zombie = mlflow.active_run()
            self.logger.warning(f"Ending lingering zombie run: {zombie.info.run_id}")
            mlflow.end_run()

        # 3. Start the Run
        # CRITICAL: Do NOT call any mlflow.log_* functions before this line.
        try:
            with mlflow.start_run(
                run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"
            ) as run:

                self.logger.info(f"MLflow Run Started. Run ID: {run.info.run_id}")

                # A. Log Context (Moved INSIDE the block)
                self._log_context(bp, run)

                # B. Load Data
                try:
                    builder = self.factory.create_dataset_builder(bp.data)
                    raw_df = builder.get_data()
                except Exception as e:
                    self.logger.error(f"Data loading failed: {e}", exc_info=True)
                    return

                if raw_df.is_empty():
                    self.logger.error("Dataset is empty. Aborting.")
                    return

                # C. Build Processor List
                processors = []
                # Use getattr to safely check for 'generate_lags' on the config object
                if getattr(bp.data, "generate_lags", False):
                    self.logger.info("Adding TemporalFeatureProcessor to pipeline.")
                    processors.append(TemporalFeatureProcessor())

                # D. Initialize Pipeline
                pipeline = HorizonPipeline(
                    model=self.factory.create_model(bp.model),
                    features=bp.data.feature_prefix_groups,
                    target=bp.data.target_column,
                    processors=processors,
                )

                # E. Preprocess
                self.logger.info("Pipeline: Preprocessing data...")
                processed_df = pipeline.preprocess(raw_df)

                # F. Extract X/y
                X, y = pipeline.get_X_y(processed_df)
                if X.empty or y is None or y.empty:
                    self.logger.error("Feature extraction failed.")
                    return

                # Log Params (Safe here because we are inside the run)
                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_param("n_features", X.shape[1])

                # G. Split
                split_idx = int(len(X) * bp.training.time_split_ratio)
                X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

                self.logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

                # H. Train
                strategy = self.factory.create_strategy(bp.training)
                trained_model = strategy.train(
                    pipeline.model, X_train, y_train, X_val, y_val
                )
                pipeline.model = trained_model

                # I. Evaluate
                evaluator = self.factory.create_evaluator(bp.training)
                metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
                mlflow.log_metrics(metrics)

                # J. Feature Importance
                self._log_feature_importance(pipeline.model, X, run)

                # K. Save
                temp_model_dir = Path("./temp_model_artifacts")
                temp_model_dir.mkdir(exist_ok=True)
                model_path = temp_model_dir / f"{bp.model_name}.pkl"

                pipeline.save(model_path)
                mlflow.log_artifact(str(model_path))
                model_path.unlink()  # Cleanup

                self.logger.success(f"Pipeline saved and logged.")

        except Exception as e:
            self.logger.exception("Training run failed")
            raise e

    def _log_context(self, bp: ModelBlueprint, run):
        """Logs metadata about the run environment."""
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
            mlflow.set_tag("git_commit", git_commit)
        except Exception:
            mlflow.set_tag("git_commit", "unknown")

        mlflow.set_tag("user", getpass.getuser())
        mlflow.set_tag("model_type", bp.model.type)

        # Flatten and log params
        from packages.ml_core.common.utils import flatten_dict

        raw_config = bp.model_dump(mode="json")
        flat_config = flatten_dict(raw_config)
        mlflow.log_params(flat_config)

    def _log_feature_importance(self, model, X_features: pd.DataFrame, run):
        """Logs feature importance if the model supports it."""
        if hasattr(model, "feature_importances_"):
            self.logger.info("Logging Feature Importance to MLflow...")

            importance_df = pd.DataFrame(
                {
                    "feature": list(X_features.columns),
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)

            # Save to a temporary CSV and upload as artifact
            imp_path = f"feature_importance_{run.info.run_id}.csv"
            importance_df.to_csv(imp_path, index=False)
            mlflow.log_artifact(imp_path)
            Path(imp_path).unlink()
