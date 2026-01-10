from typing import Protocol
import pandas as pd
import polars as pl
from packages.contracts.blueprints import DataConfigType


class ValidationLogic(Protocol):
    """
    Pure logic component for validation steps.
    Must be pickle-safe (no complex object state).
    """

    def permute_data(self, df: pd.DataFrame, seed: int) -> pd.DataFrame: ...

    def relabel_data(
        self, df: pl.DataFrame, config: DataConfigType
    ) -> pl.DataFrame: ...


class ModelLifecycle(Protocol):
    """
    The Manager for a specific model type (Regime vs Alpha).
    Handles Data Prep and providing Validation Logic.
    """

    def prepare_training_data(
        self, raw_df: pl.DataFrame, config: DataConfigType
    ) -> pl.DataFrame:
        """
        Main pipeline entry: Raw DB Data -> Features -> Labels -> Ready for Training.
        """
        ...

    def get_validation_logic(self) -> ValidationLogic:
        """
        Returns the worker logic for Monte Carlo / Stability tests.
        """
        ...
