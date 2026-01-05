import joblib
import re
from pathlib import Path
from typing import Any, Dict, List
import polars as pl
import mlflow.pyfunc
import pandas as pd

from packages.ml_core.data.processors.base import BaseProcessor

# Import LogManager to create loggers on the fly (Pickle-safe)
from packages.quant_lib.logging import LogManager


class HorizonPipeline:
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

        self.trained_features: List[str] | None = None
        self.run_id: str | None = None
        self.metadata: Dict[str, Any] = {}

    @property
    def features(self) -> List[str]:
        return self.trained_features if self.trained_features else self.feature_prefixes

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. Run Processors (Add Lags, etc)
        for proc in self.processors:
            df = proc.transform(df)

        # 2. Enforce Float Types (for Int columns that aren't target)
        int_cols = [
            col
            for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
            and col != self.target_col
        ]
        if int_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in int_cols])

        # 3. Clean Rows
        if self.target_col in df.columns:
            df = df.drop_nulls(subset=[self.target_col])

        return df

    def get_X_y(self, df: pl.DataFrame):
        # Instantiate logger here to avoid pickling issues
        logger = LogManager("horizon-pipeline", debug=True).get_logger("selector")

        all_cols = df.columns

        if self.trained_features:
            # Production Mode: Exact Match
            selected_features = [c for c in self.trained_features if c in all_cols]
        else:
            # Training Mode: Discovery

            # A. Include based on prefixes
            included = {
                c
                for c in all_cols
                if any(c.startswith(p) for p in self.feature_prefixes)
            }

            # LOG CANDIDATES
            # logger.info(f"Candidates (matched prefixes): {sorted(list(included))}")

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

            if excluded:
                logger.warning(f"✂️  Dropped {len(excluded)} features via exclusion:")
                logger.warning(f"    {sorted(list(excluded))}")

            selected_features = sorted(list(included - excluded))

            # LOG FINAL
            logger.info(f"✅ Final Feature Selection ({len(selected_features)}):")
            logger.info(f"    {selected_features}")

        if not selected_features:
            raise ValueError(
                f"No feature columns selected! Prefixes: {self.feature_prefixes}, Excluded: {self.exclude_patterns}"
            )

        X = df.select(selected_features).to_pandas()

        y = None
        if self.target_col in df.columns:
            y = df.select(self.target_col).to_pandas()

        return X, y

    def predict(self, df: pl.DataFrame):
        for proc in self.processors:
            df = proc.transform(df)
        X, _ = self.get_X_y(df)
        return self.model.predict(X)

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
