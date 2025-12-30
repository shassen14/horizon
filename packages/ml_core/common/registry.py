import mlflow
from typing import Optional
from packages.ml_core.modeling.pipeline import HorizonPipeline


class ModelRegistryClient:
    """
    Abstracts the retrieval of trained models.
    Currently backed by MLflow, but could easily swap to S3 or Local Disk.
    """

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        # Initialize immediately
        mlflow.set_tracking_uri(self.tracking_uri)

    def load_pipeline(
        self, model_name: str, alias: str = "production"
    ) -> Optional[HorizonPipeline]:
        """
        Loads a HorizonPipeline from the registry.
        Returns None if not found, instead of crashing.
        """
        uri = f"models:/{model_name}@{alias}"

        try:
            # Load the generic PyFunc wrapper
            wrapper = mlflow.pyfunc.load_model(uri)

            # Unwrap our custom HorizonPipeline object
            # This relies on the HorizonMLflowWrapper implementation we wrote
            pipeline = wrapper._model_impl.python_model.pipeline
            return pipeline

        except Exception as e:
            # We log internally or just return None to let caller decide
            print(f"Registry Warning: Could not load {uri}: {e}")
            return None
