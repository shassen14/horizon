# packages/ml_core/pipeline.py

import joblib
from pathlib import Path
from typing import Any, List
import polars as pl
import mlflow.pyfunc
import pandas as pd

from packages.ml_core.data.processors.base import BaseProcessor


class HorizonPipeline:
    def __init__(
        self,
        model: Any,
        features: List[str],
        target: str,
        processors: List[BaseProcessor] = None,
    ):
        self.model = model
        self.target_col = target
        self.feature_cols = (
            features  # The list of prefixes to keep (e.g. ["sma_", "rsi_"])
        )

        # The Pipeline owns the preprocessor
        self.processors = processors if processors else []

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Runs the data through the chain of processors sequentially.
        """
        # 1. Run chain
        for proc in self.processors:
            # Optional: Add logging here for observability
            # print(f"Running processor: {type(proc).__name__}")
            df = proc.transform(df)

        # 2. Cleaning (Drop warmup rows caused by lags)
        # Note: We might want to make 'DropNulls' its own explicit processor later
        # But for now, keeping it here as a safety net is fine.
        df = df.drop_nulls()

        return df

    def get_X_y(self, df: pl.DataFrame):
        """
        Extracts features (X) and target (y) from a processed DataFrame.
        """
        # Select columns matching the configured prefixes
        all_cols = df.columns
        selected_features = [
            c for c in all_cols if any(c.startswith(p) for p in self.feature_cols)
        ]

        X = df.select(selected_features).to_pandas()

        # Handle Target (might not exist during live inference)
        y = None
        if self.target_col in df.columns:
            y = df.select(self.target_col).to_pandas()

        return X, y

    def predict(self, df: pl.DataFrame):
        """
        End-to-End Inference: Raw Data -> Prediction
        """
        # 1. Transform (Add Lags)
        # Note: We do NOT drop nulls here, because we want to predict
        # on the last row even if previous history is partial,
        # assuming the preprocessor handles it safely (or we trim carefully).
        for proc in self.processors:
            df = proc.transform(df)

        # 2. Extract X
        X, _ = self.get_X_y(df)

        # 3. Predict
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


class HorizonMLflowWrapper(mlflow.pyfunc.PythonModel):
    """
    Adapter to make HorizonPipeline compatible with MLflow Registry.
    """

    def __init__(self, pipeline: HorizonPipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame):
        """
        MLflow standard inference method.
        model_input: Usually a Pandas DataFrame coming from the API or UI.
        """
        # 1. Convert Input to Polars (if it isn't already)
        if isinstance(model_input, pd.DataFrame):
            df = pl.from_pandas(model_input)
        else:
            df = model_input

        # 2. Delegate to the actual pipeline
        return self.pipeline.predict(df)
