import numpy as np
import pandas as pd
from pathlib import Path

from packages.ml_ops.artifacts import TrainingArtifacts
from packages.ml_ops.tracker import ExperimentTracker
from .base import BaseValidator, ValidationResult
from packages.ml_ops.training.factory import MLComponentFactory
from packages.contracts.blueprints import TrainingConfig


class AblationValidator(BaseValidator):
    def __init__(self, logger, factory: MLComponentFactory, config: TrainingConfig):
        super().__init__(logger)
        self.factory = factory
        self.config = config

    def validate(
        self, artifacts: TrainingArtifacts, tracker: ExperimentTracker
    ) -> ValidationResult:
        self.logger.info("--- Running Feature Ablation ---")

        model = artifacts.pipeline.model
        X_val, y_val = artifacts.X_val, artifacts.y_val
        evaluator = self.factory.create_evaluator(self.config)
        metric_name = self.config.eval_metric
        base_score = artifacts.metrics.get(metric_name, 0)

        results = []
        for feat in list(X_val.columns):
            X_corrupted = X_val.copy()
            X_corrupted[feat] = np.random.permutation(X_corrupted[feat].values)

            c_metrics = evaluator.evaluate(model, X_corrupted, y_val, logger=None)
            c_score = c_metrics.get(metric_name, base_score)

            results.append({"feature": feat, "impact": abs(c_score - base_score)})

        df_res = pd.DataFrame(results).sort_values("impact", ascending=False)

        csv_name = "feature_ablation.csv"
        df_res.to_csv(csv_name, index=False)
        tracker.log_artifact(csv_name)
        Path(csv_name).unlink()

        top_feat = df_res.iloc[0]["feature"] if not df_res.empty else "None"

        return ValidationResult(
            name="Ablation",
            passed=True,  # Ablation is informational, it doesn't fail a build
            details={"top_feature_by_impact": top_feat},
        )
