# packages/ml_core/evaluation/base.py

from abc import ABC, abstractmethod
import pandas as pd


class EvaluationStrategy(ABC):
    """Abstract Base Class for all model evaluation procedures."""

    @abstractmethod
    def evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, logger
    ) -> dict:
        """
        Takes a trained model and test data, performs evaluation, logs results,
        and returns a dictionary of key metrics.
        """
        pass
