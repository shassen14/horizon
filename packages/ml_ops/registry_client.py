import mlflow
import pandas as pd
import polars as pl
from typing import Optional
from packages.contracts.vocabulary.columns import MarketCol
from packages.ml_ops.modeling.pipeline import HorizonPipeline, HorizonMLflowWrapper
from packages.contracts.blueprints import ModelBlueprint
from packages.ml_ops.artifacts import TrainingArtifacts
from packages.ml_ops.tracker import ExperimentTracker


class RegistryClient:
    """
    Unified client for interacting with the MLflow Model Registry.
    Handles both loading (reading) and registering (writing) models.
    """

    def __init__(self, tracking_uri: str, logger=None):
        self.tracking_uri = tracking_uri
        self.logger = logger
        mlflow.set_tracking_uri(self.tracking_uri)

    # --- READ METHOD ---
    def load_pipeline(
        self, model_name: str, alias: str = "production"
    ) -> Optional[HorizonPipeline]:
        """Loads a trained HorizonPipeline from the MLflow registry."""
        uri = f"models:/{model_name}@{alias}"

        if self.logger:
            self.logger.info(f"Loading model from registry: {uri}")

        try:
            wrapper = mlflow.pyfunc.load_model(uri)
            pipeline = wrapper._model_impl.python_model.pipeline

            if isinstance(pipeline, HorizonPipeline):
                if self.logger:
                    self.logger.success(
                        f"Successfully loaded pipeline for '{model_name}'."
                    )
                return pipeline

            if self.logger:
                self.logger.error(
                    f"Failed to unwrap HorizonPipeline from loaded model '{model_name}'."
                )
            return None

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Could not load model '{uri}' from registry: {e}", exc_info=True
                )
            return None

    # --- WRITE METHOD ---
    def register_pipeline(
        self,
        blueprint: ModelBlueprint,
        artifacts: TrainingArtifacts,
        tracker: ExperimentTracker,
    ) -> str:
        """
        Wraps, signs, and registers a certified pipeline to the MLflow Model Registry.
        """
        pipeline = artifacts.pipeline
        input_example = artifacts.input_example

        if self.logger:
            self.logger.info(f"Registering model '{blueprint.model_name}' to MLflow...")

        time_col = MarketCol.TIME
        if time_col in input_example.columns and pd.api.types.is_datetime64_any_dtype(
            input_example[time_col]
        ):
            if input_example[time_col].dt.tz is not None:
                self.logger.info(
                    "Converting timezone-aware datetime to naive UTC for MLflow signature."
                )
                input_example = input_example.copy()  # Avoid SettingWithCopyWarning
                input_example[time_col] = input_example[time_col].dt.tz_localize(None)

        # 1. Handle int types for safety
        int_cols = input_example.select_dtypes(include=["int"]).columns
        if len(int_cols) > 0:
            input_example = input_example.copy()
            input_example[int_cols] = input_example[int_cols].astype(float)

        prediction_example = pipeline.predict(pl.from_pandas(input_example))
        signature = tracker.log_model_signature(input_example, prediction_example)

        # 2. Wrap for pyfunc standard
        mlflow_model = HorizonMLflowWrapper(pipeline)

        # 3. Log and Register
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=mlflow_model,
            registered_model_name=blueprint.model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=blueprint.model.dependencies,
        )

        return model_info.registered_model_version
