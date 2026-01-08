# packages/data_pipelines/labelers/base.py

from abc import ABC, abstractmethod
import polars as pl


class AbstractLabeler(ABC):
    """
    Abstract contract for any object that can take a DataFrame
    and return it with an added 'label' column.
    """

    @abstractmethod
    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Takes a DataFrame, applies some logic, and returns the DataFrame
        with an additional column (e.g., 'regime').
        """
        pass
