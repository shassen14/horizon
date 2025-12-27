# packages/ml_core/evaluation/regression.py

import pandas as pd
from scipy.stats import spearmanr

from packages.ml_core.evaluation.base import EvaluationStrategy


class RegressionEvaluator(EvaluationStrategy):
    """
    Evaluation strategy for regression models, focusing on ranking performance.
    Calculates Spearman Correlation and Quintile Analysis.
    """

    def evaluate(self, model, X_test, y_test, logger) -> dict:
        logger.info("--- Using RegressionEvaluator ---")

        # 1. Get Predictions
        predictions = model.predict(X_test)

        # 2. Calculate Spearman Rank Correlation
        spearman_corr, _ = spearmanr(y_test, predictions)
        logger.info(f"Spearman Rank Correlation (rho): {spearman_corr:.4f}")

        if spearman_corr > 0.02:
            logger.success("✅ Model shows a positive ranking edge.")
        else:
            logger.warning("❌ Model ranking edge is weak or negative.")

        # 3. Perform Quintile Analysis
        eval_df = pd.DataFrame({"true": y_test.iloc[:, 0], "pred": predictions})

        quintile_threshold = eval_df["pred"].quantile(0.8)
        top_quintile_stocks = eval_df[eval_df["pred"] >= quintile_threshold]

        avg_return_top = top_quintile_stocks["true"].mean()
        avg_return_all = eval_df["true"].mean()

        logger.info(f"Avg 3-Mo Return (All): {avg_return_all:.4f}")
        logger.info(f"Avg 3-Mo Return (Top 20%): {avg_return_top:.4f}")

        if avg_return_top > avg_return_all:
            logger.success("✅ Top Quintile portfolio created alpha.")
        else:
            logger.warning("❌ Top Quintile portfolio did NOT outperform.")

        # Return key metrics for logging or comparison
        return {
            "spearman_rho": spearman_corr,
            "top_quintile_return": avg_return_top,
            "average_return": avg_return_all,
        }
