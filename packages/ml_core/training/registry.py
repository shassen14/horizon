import mlflow
import polars as pl
from packages.ml_core.modeling.pipeline import HorizonMLflowWrapper


class ModelRegistrar:
    def __init__(self, logger):
        self.logger = logger

    def register(self, blueprint, artifacts, tracker):
        pipeline = artifacts.pipeline
        X_val = artifacts.X_val

        # 1. Infer Signature
        input_example = X_val.head(5)
        int_cols = input_example.select_dtypes(include=["int"]).columns
        if len(int_cols) > 0:
            input_example = input_example.copy()
            input_example[int_cols] = input_example[int_cols].astype(float)

        prediction_example = pipeline.predict(pl.from_pandas(input_example))
        signature = tracker.log_model_signature(input_example, prediction_example)

        # 2. Log Model
        mlflow_model = HorizonMLflowWrapper(pipeline)
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=mlflow_model,
            registered_model_name=blueprint.model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=blueprint.model.dependencies,
        )
        return model_info.registered_model_version
