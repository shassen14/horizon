# packages/ml_core/data/processors/base.py

from abc import ABC, abstractmethod
import polars as pl


class BaseProcessor(ABC):
    """
    Abstract contract for any data transformation step.
    Must accept a Polars DataFrame and return a transformed Polars DataFrame.
    """

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    # Optional: fit method if the processor needs to learn state (like Mean/Std for scaling)
    # def fit(self, df: pl.DataFrame):
    #     return self
