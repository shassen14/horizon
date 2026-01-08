import joblib
import re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import polars as pl
import mlflow.pyfunc
import pandas as pd

from packages.contracts.vocabulary.columns import MarketCol
from packages.data_pipelines.processors.base import BaseProcessor
from packages.quant_lib.logging import LogManager


class HorizonPipeline:
    """
    The core serializable object for a trained model.
    Encapsulates preprocessing, feature selection, and the trained model instance.
    This object is what gets saved to MLflow and loaded for inference.
    """

    def __init__(
        self,
        model: Any,
        feature_prefixes: List[str],
        target: str,
        processors: List[BaseProcessor] = None,
        exclude_patterns: List[str] = None,
    ):
        self.model = model
        self.target_col = target
        self.feature_prefixes = feature_prefixes
        self.exclude_patterns = exclude_patterns if exclude_patterns else []
        self.processors = processors if processors else []

        # This will be "frozen" after the first call to get_X_y during training
        self.trained_features: List[str] | None = None

        # Metadata for traceability
        self.run_id: str | None = None
        self.metadata: Dict[str, Any] = {}

    @property
    def features(self) -> List[str]:
        return self.trained_features if self.trained_features else self.feature_prefixes

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Runs the full preprocessing chain: processors, type enforcement, and null cleaning.
        Returns a "fat" DataFrame ready for feature selection.
        """
        # 1. Run Transformation Processors (e.g., TemporalFeatureProcessor for lags)
        for proc in self.processors:
            df = proc.transform(df)

        # 2. Enforce Float Types for ML compatibility
        int_cols = [
            col
            for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
            and col != self.target_col
        ]
        if int_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in int_cols])

        # 3. Clean Rows with Missing Targets (useless for training)
        if self.target_col in df.columns:
            df = df.drop_nulls(subset=[self.target_col])

        return df

    def get_X_y(self, df: pl.DataFrame):
        """
        Extracts features (X) and target (y) from a processed DataFrame.
        This is where feature selection logic is applied.
        """
        # Instantiate logger here to avoid pickling issues
        logger = LogManager("horizon-pipeline", debug=True).get_logger("selector")

        if self.trained_features:
            # --- INFERENCE MODE ---
            # Use the exact, frozen list of columns from training time.
            selected_features = [c for c in self.trained_features if c in df.columns]
        else:
            # --- TRAINING MODE ---
            # Discover features based on prefixes and exclusions.
            all_cols = df.columns

            # A. Include based on prefixes
            included = {
                c
                for c in all_cols
                if any(c.startswith(p) for p in self.feature_prefixes)
            }

            # LOG CANDIDATES
            logger.info(f"Candidates (matched prefixes): {sorted(list(included))}")

            # B. Exclude based on patterns
            excluded = set()
            if self.exclude_patterns:
                patterns = []
                for p in self.exclude_patterns:

                    # CASE 1: Pure Numbers (e.g. "50", "63")
                    # We use strict boundaries to prevent "20" killing "200"
                    if p.isdigit():
                        # Matches "_60" or "_60_" (Snake case suffix)
                        patterns.append(re.compile(rf".*_{p}($|_)"))
                        # Matches "sma50" or "sma50_" (Letter-Number suffix)
                        patterns.append(re.compile(rf".*[a-z]{p}($|_)"))

                    # CASE 2: Named Patterns (e.g. "return_5", "adx_")
                    # We use aggressive substring matching
                    else:
                        patterns.append(re.compile(rf".*{re.escape(p)}.*"))

                for col in included:
                    if any(pat.match(col) for pat in patterns):
                        excluded.add(col)

            selected_features = sorted(list(included - excluded))

            # Freeze the list for future inference calls
            self.trained_features = selected_features

            # Logging for traceability
            logger.info(
                f"Feature selection complete. Freezing {len(selected_features)} features for this pipeline."
            )
            # if excluded:
            #     logger.warning(
            #         f"  -> Dropped {len(excluded)} features via exclusion: {sorted(list(excluded))}"
            #     )
            logger.info(f"  -> Final features: {selected_features}")

        if not selected_features:
            raise ValueError(
                "No feature columns were selected. Check your configuration."
            )

        X = df.select(selected_features).to_pandas()

        y = None
        if self.target_col in df.columns:
            y = df.select(self.target_col).to_pandas()

        return X, y

    def predict(self, df: pl.DataFrame):
        """End-to-End Inference: Raw Data -> Prediction"""
        # The input df is assumed to have the necessary raw columns for preprocessing
        processed_df = self.preprocess(df)
        X, _ = self.get_X_y(processed_df)

        # Drop rows with any NaNs in the selected features before prediction
        # LightGBM/XGBoost can handle this, but sklearn models cannot
        X_clean = X.dropna()
        if len(X_clean) == 0:
            return np.array([])  # Return empty if no valid rows

        preds = self.model.predict(X_clean)

        # We need to return a result that aligns with the original input index
        # This creates a pandas Series with the original index and predictions
        result_series = pd.Series(preds, index=X_clean.index)

        # Reindex to match the full input DataFrame's index, filling missing with NaN
        return result_series.reindex(df.to_pandas().index).to_numpy()

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


class HorizonMLflowWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline: HorizonPipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame):
        if isinstance(model_input, pd.DataFrame):
            df = pl.from_pandas(model_input)
        else:
            df = model_input
        return self.pipeline.predict(df)
