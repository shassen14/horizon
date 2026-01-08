from abc import ABC, abstractmethod
import pandas as pd


class BasePermutator(ABC):
    """
    Abstract contract for any object that can create a statistically similar,
    but temporally random, version of a time-series DataFrame.
    """

    @abstractmethod
    def permute(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """
        Takes a raw DataFrame, scrambles it according to the strategy,
        and returns a synthetic DataFrame.
        """
        pass
