from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class EvaluationStrategy(ABC):
    """
    Abstract Base Class for all model evaluation procedures.
    Its job is to take a trained model and test data, and return a dictionary of metrics.
    """

    @abstractmethod
    def evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, logger
    ) -> Dict[str, float]:
        """
        Takes a trained model and test data, performs evaluation, logs results,
        and returns a dictionary of key metrics.

        Args:
            model: The trained model object with a .predict() method.
            X_test (pd.DataFrame): The feature set for evaluation.
            y_test (pd.DataFrame): The ground truth labels or values.
            logger: A configured logger instance for printing results.

        Returns:
            Dict[str, float]: A dictionary of metric names and their calculated values.
        """
        pass
