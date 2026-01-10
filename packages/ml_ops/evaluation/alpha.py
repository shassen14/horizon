import pandas as pd
from scipy.stats import spearmanr
from .base import EvaluationStrategy
from typing import Dict


class AlphaEvaluator(EvaluationStrategy):
    """
    Evaluation strategy for Alpha (Ranking) models.
    Primary metric is the Information Coefficient (Spearman Rank Correlation).
    """

    def evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, logger
    ) -> Dict[str, float]:
        if logger:
            logger.info("--- Using AlphaEvaluator ---")

        predictions = model.predict(X_test)

        # Create a DataFrame for easy correlation calculation
        # Drop NaNs that may result from prediction alignment
        eval_df = pd.DataFrame(
            {"prediction": predictions, "actual": y_test.values.ravel()}
        ).dropna()

        if len(eval_df) < 10:
            if logger:
                logger.warning("Not enough valid samples to calculate IC.")
            return {"information_coefficient": 0.0, "p_value": 1.0}

        # Calculate Spearman Rank Correlation (Information Coefficient)
        ic, p_value = spearmanr(eval_df["prediction"], eval_df["actual"])

        # Handle NaN case if correlation is perfect (std dev is zero)
        ic = 0.0 if pd.isna(ic) else ic

        if logger:
            logger.info(f"  Information Coefficient (IC): {ic:.4f}")
            logger.info(f"  P-value:                    {p_value:.4f}")
            if ic > 0.02:
                logger.success(
                    "  -> Model shows a statistically significant ranking ability."
                )
            else:
                logger.warning("  -> Model's ranking ability is weak or non-existent.")

        return {"information_coefficient": ic, "p_value": p_value}
