# packages/ml_core/common/artifacts.py

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import polars as pl
from packages.ml_ops.modeling.pipeline import HorizonPipeline


@dataclass
class TrainingArtifacts:
    """
    Holds the state of a training run.
    Passed between Trainer -> Validators -> Registrar.
    """

    pipeline: HorizonPipeline
    raw_df: pl.DataFrame
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame

    # Metadata
    metrics: Dict[str, float]
    feature_names: List[str]
    cache_key: str  # For tagging the dataset used
