# packages/ml_core/validation/analyzers.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict


class StabilityAnalyzer:
    """
    Injects noise into X_val and measures how often predictions flip.
    """

    def __init__(self, logger):
        self.logger = logger

    def run(
        self, model, X_val: pd.DataFrame, noise_levels: List[float] = [0.01, 0.05, 0.10]
    ) -> pd.DataFrame:
        self.logger.info("--- Running Stability Analysis ---")

        base_preds = model.predict(X_val)
        X_std = X_val.std()
        results = []

        for noise_pct in noise_levels:
            # Generate Noise
            noise_matrix = np.random.normal(
                loc=0.0, scale=X_std * noise_pct, size=X_val.shape
            )
            X_noisy = X_val + noise_matrix

            # Predict
            noisy_preds = model.predict(X_noisy)

            # Measure
            flips = np.sum(base_preds != noisy_preds)
            flip_rate = flips / len(base_preds)

            results.append({"noise_pct": noise_pct, "flip_rate": flip_rate})

        return pd.DataFrame(results)


class AblationAnalyzer:
    """
    Performs Permutation Importance on a trained model.
    """

    def __init__(self, logger, evaluator, primary_metric: str):
        self.logger = logger
        self.evaluator = evaluator
        self.primary_metric = primary_metric

    def run(
        self, model, X_val: pd.DataFrame, y_val: pd.DataFrame, baseline_metrics: Dict
    ) -> pd.DataFrame:
        self.logger.info("--- Running Feature Ablation ---")

        base_score = baseline_metrics.get(
            self.primary_metric, list(baseline_metrics.values())[0]
        )
        feature_names = list(X_val.columns)
        results = []

        for i, feat in enumerate(feature_names):
            if i % 5 == 0:
                self.logger.info(f"   Ablating {feat}...")

            # Shuffle
            X_corrupted = X_val.copy()
            X_corrupted[feat] = np.random.permutation(X_corrupted[feat].values)

            # Evaluate (Silence the evaluator logger)
            c_metrics = self.evaluator.evaluate(model, X_corrupted, y_val, logger=None)
            c_score = c_metrics.get(self.primary_metric, list(c_metrics.values())[0])

            # Calculate Delta (Magnitude of change)
            delta = abs(c_score - base_score)

            results.append(
                {
                    "feature": feat,
                    "baseline": base_score,
                    "corrupted": c_score,
                    "delta": delta,
                }
            )

        return pd.DataFrame(results).sort_values("delta", ascending=False)
