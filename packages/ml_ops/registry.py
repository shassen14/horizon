import mlflow
from typing import Optional
from packages.ml_ops.modeling.pipeline import HorizonPipeline


class ModelRegistryClient:
    """
    Client for interacting with the MLflow Model Registry.
    Primary responsibility: Loading trained HorizonPipeline objects.
    """

    def __init__(self, tracking_uri: str, logger=None):
        self.tracking_uri = tracking_uri
        self.logger = logger
        # Set the tracking URI for all MLflow operations in this context
        mlflow.set_tracking_uri(self.tracking_uri)

    def load_pipeline(
        self, model_name: str, alias: str = "production"
    ) -> Optional[HorizonPipeline]:
        """
        Loads a trained HorizonPipeline from the MLflow registry using a model name and alias/version.

        Args:
            model_name (str): The registered name of the model.
            alias (str): The alias (e.g., "production", "staging") or version number.

        Returns:
            Optional[HorizonPipeline]: The loaded pipeline object, or None if loading fails.
        """
        uri = f"models:/{model_name}@{alias}"

        if self.logger:
            self.logger.info(f"Loading model from registry: {uri}")

        try:
            # mlflow.pyfunc.load_model loads our custom HorizonMLflowWrapper
            wrapper = mlflow.pyfunc.load_model(uri)

            # We need to access the underlying custom pipeline object.
            # This relies on the structure defined in HorizonMLflowWrapper.
            # The path is: wrapper -> _model_impl -> python_model -> pipeline
            if hasattr(wrapper, "_model_impl") and hasattr(
                wrapper._model_impl, "python_model"
            ):
                pipeline = wrapper._model_impl.python_model.pipeline
                if isinstance(pipeline, HorizonPipeline):
                    if self.logger:
                        self.logger.success(
                            f"Successfully loaded pipeline for '{model_name}'."
                        )
                    return pipeline

            # If the structure is not as expected
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
