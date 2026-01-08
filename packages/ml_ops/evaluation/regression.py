import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from .base import EvaluationStrategy
from typing import Dict


class RegressionEvaluator(EvaluationStrategy):
    """
    Evaluation strategy for regression models.
    Calculates key error metrics and goodness-of-fit.
    """

    def evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, logger
    ) -> Dict[str, float]:
        if logger:
            logger.info("--- Using RegressionEvaluator ---")

        # 1. Get Predictions
        predictions = model.predict(X_test)

        # 2. Calculate Metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # 3. Logging (if a logger is provided)
        if logger:
            logger.info(f"  Mean Squared Error (MSE):      {mse:.6f}")
            logger.info(f"  Root Mean Squared Error (RMSE):{rmse:.6f}")
            logger.info(f"  Mean Absolute Error (MAE):     {mae:.6f}")
            logger.info(f"  R-squared (R²):                {r2:.4f}")

            if r2 > 0.01:
                logger.success("  -> Model shows a positive predictive correlation.")
            elif r2 < 0:
                logger.error("  -> Model performs worse than a simple mean forecast.")
            else:
                logger.warning("  -> Model performance is close to zero correlation.")

        # 4. Return structured dictionary
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
