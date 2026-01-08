# packages/ml_core/data/processors/temporal.py

import polars as pl
from typing import List

from packages.data_pipelines.processors.base import BaseProcessor


class TemporalFeatureProcessor(BaseProcessor):
    """
    Transforms a DataFrame of base features into a model-ready feature set
    by adding temporal context (lags and deltas).
    """

    def __init__(self):
        self.features_to_lag: List[str] = [
            "rsi_14",
            "relative_volume",
            "atr_14_pct",
            "return_1",
            "zscore_20",
        ]
        self.lag_periods: List[int] = [1, 3, 5, 21, 63]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Main entry point. Adds lag and delta columns to the DataFrame.
        """
        if df.is_empty():
            return df

        group_by_col = "asset_id" if "asset_id" in df.columns else None

        lag_expressions = self._generate_lag_expressions()

        if group_by_col:
            # Apply window functions within each asset group
            df = df.with_columns([expr.over(group_by_col) for expr in lag_expressions])
        else:
            # Apply directly if it's a single asset
            df = df.with_columns(lag_expressions)

        return df

    def _generate_lag_expressions(self) -> List[pl.Expr]:
        """
        Creates a list of Polars expressions for all configured lags and deltas.
        """
        expressions = []

        for feature in self.features_to_lag:
            for lag in self.lag_periods:
                # 1. Lag Feature
                lag_expr = pl.col(feature).shift(lag).alias(f"{feature}_lag_{lag}")
                expressions.append(lag_expr)

                # 2. Delta Feature
                delta_expr = (pl.col(feature) - pl.col(feature).shift(lag)).alias(
                    f"{feature}_delta_{lag}"
                )
                expressions.append(delta_expr)

        return expressions
