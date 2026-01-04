# packages/ml_core/pipeline.py

import joblib
from pathlib import Path
from typing import Any, Dict, List
import polars as pl
import mlflow.pyfunc
import pandas as pd

from packages.ml_core.data.processors.base import BaseProcessor


class HorizonPipeline:
    def __init__(
        self,
        model: Any,
        feature_prefixes: List[str],
        target: str,
        processors: List[BaseProcessor] = None,
    ):
        self.model = model
        self.target_col = target
        self.feature_prefixes = (
            feature_prefixes  # The list of prefixes to keep (e.g. ["sma_", "rsi_"])
        )
        self.processors = processors if processors else []
        self.trained_features: List[str] | None = None
        self.run_id: str | None = None
        self.metadata: Dict[str, Any] = {}

    @property
    def features(self) -> List[str]:
        """Alias for feature_cols to allow easier access."""
        return self.trained_features if self.trained_features else self.feature_prefixes

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Runs the data through the chain of processors sequentially.
        """
        # 1. Run chain
        for proc in self.processors:
            # Optional: Add logging here for observability
            # print(f"Running processor: {type(proc).__name__}")
            df = proc.transform(df)

        # Resolve features: use trained features if available, else prefixes
        if self.trained_features:
            relevant_cols = [c for c in self.trained_features if c in df.columns]
        else:
            # Discovery mode: Find columns matching prefixes
            relevant_cols = [
                c
                for c in df.columns
                if any(c.startswith(p) for p in self.feature_prefixes)
            ]

        # Add target to the check list if it exists
        if self.target_col in df.columns:
            relevant_cols.append(self.target_col)

        if relevant_cols:
            df = df.drop_nulls(subset=relevant_cols)
        else:
            # Fallback if no cols matched (shouldn't happen)
            df = df.drop_nulls()

        # 3. Enforce Float Types
        # We convert all Integers to Floats.
        # This ensures MLflow Signatures allow NaNs/Nulls during inference validation.
        # It also ensures consistency between Train (which logged Float) and Serve.
        int_cols = [
            col
            for col, dtype in zip(df.columns, df.dtypes)
            if dtype
            in (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            )
        ]

        if int_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in int_cols])

        return df

    def get_X_y(self, df: pl.DataFrame):
        """
        Extracts features (X) and target (y) from a processed DataFrame.
        """
        # Select columns matching the configured prefixes
        all_cols = df.columns

        if self.trained_features:
            # Strict Mode: We know exactly what columns we need.
            # Check if they exist
            missing = [c for c in self.trained_features if c not in all_cols]
            if missing:
                # In production, this might raise an error. For now, print warning.
                print(
                    f"Warning: Pipeline expects columns {missing} which are missing from input."
                )

            selected_features = [c for c in self.trained_features if c in all_cols]
        else:
            # Discovery Mode (First run / Training time): Use prefixes
            selected_features = [
                c
                for c in all_cols
                if any(c.startswith(p) for p in self.feature_prefixes)
            ]

        # Sort to ensure column order is always consistent for the model
        selected_features = sorted(selected_features)

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
