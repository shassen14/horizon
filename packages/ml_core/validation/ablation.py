import numpy as np
import pandas as pd
from pathlib import Path
from packages.ml_core.common.artifacts import TrainingArtifacts
from packages.ml_core.common.tracker import ExperimentTracker
from packages.ml_core.validation.base import BaseValidator, ValidationResult
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.common.schemas import TrainingConfig


class AblationValidator(BaseValidator):
    def __init__(self, logger, factory: MLComponentFactory, config: TrainingConfig):
        super().__init__(logger)
        self.factory = factory
        self.config = config

    def validate(
        self, artifacts: TrainingArtifacts, tracker: ExperimentTracker
    ) -> ValidationResult:
        self.logger.info("Running Feature Ablation...")

        model = artifacts.pipeline.model
        X_val = artifacts.X_val
        y_val = artifacts.y_val

        # Create Evaluator from Factory
        evaluator = self.factory.create_evaluator(self.config)

        # Baseline
        metric_name = self.config.eval_metric
        base_score = artifacts.metrics.get(metric_name)

        if base_score is None:
            metric_name = list(artifacts.metrics.keys())[0]
            base_score = artifacts.metrics[metric_name]

        feature_names = list(X_val.columns)
        results = []

        for i, feat in enumerate(feature_names):
            # Log progress every 20%
            if len(feature_names) > 5 and i % (len(feature_names) // 5) == 0:
                self.logger.info(f"   Ablating {feat} ({i+1}/{len(feature_names)})...")

            X_corrupted = X_val.copy()
            X_corrupted[feat] = np.random.permutation(X_corrupted[feat].values)

            c_metrics = evaluator.evaluate(model, X_corrupted, y_val, logger=None)
            c_score = c_metrics.get(metric_name)

            if c_score is None:
                continue

            impact = abs(c_score - base_score)
            delta = c_score - base_score

            results.append(
                {
                    "feature": feat,
                    "baseline": base_score,
                    "corrupted": c_score,
                    "impact": impact,
                    "delta": delta,
                }
            )

        df_res = pd.DataFrame(results).sort_values("impact", ascending=False)

        csv_name = "feature_ablation.csv"
        df_res.to_csv(csv_name, index=False)
        tracker.log_artifact(csv_name)
        Path(csv_name).unlink()

        top_feat = df_res.iloc[0]["feature"] if not df_res.empty else "None"

        return ValidationResult(
            name="Ablation", passed=True, details={"top_driver": top_feat}
        )
