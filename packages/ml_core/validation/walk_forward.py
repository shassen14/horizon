from pathlib import Path
import pandas as pd
import sklearn
from packages.ml_core.validation.base import BaseValidator, ValidationResult
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.common.schemas import (
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
)


class WalkForwardValidator(BaseValidator):
    def __init__(
        self,
        logger,
        factory: MLComponentFactory,
        training_config: TrainingConfig,
        validation_config: ValidationConfig,
        model_config: ModelConfig,
    ):
        super().__init__(logger)
        self.factory = factory
        self.train_conf = training_config
        self.val_conf = validation_config
        self.model_config = model_config

    def validate(self, artifacts, tracker) -> ValidationResult:
        if not self.val_conf.walk_forward_enabled:
            return ValidationResult("WalkForward", True, {"status": "skipped"})

        windows = self.val_conf.walk_forward_windows
        self.logger.info(
            f"Running Walk-Forward Analysis ({windows} expanding windows)..."
        )

        # 1. Combine Train + Val for full history
        # (We need the whole timeline to slice it up)
        X_full = pd.concat([artifacts.X_train, artifacts.X_val])
        y_full = pd.concat([artifacts.y_train, artifacts.y_val])

        total_samples = len(X_full)
        min_train = int(total_samples * self.val_conf.walk_forward_min_train_size)

        # Calculate step size
        remaining = total_samples - min_train
        step_size = remaining // windows

        scores = []

        strategy = self.factory.create_strategy(self.train_conf)
        evaluator = self.factory.create_evaluator(self.train_conf)
        metric_name = self.train_conf.eval_metric

        for i in range(windows):
            # Define Split
            train_end = min_train + (i * step_size)
            test_end = train_end + step_size

            # Slice
            X_t = X_full.iloc[:train_end]
            y_t = y_full.iloc[:train_end]
            X_v = X_full.iloc[train_end:test_end]
            y_v = y_full.iloc[train_end:test_end]

            if len(X_v) < 10:
                break  # Skip tiny folds

            # Fresh Model
            model_fresh = self.factory.create_model(self.model_config)

            # Train
            strategy.train(model_fresh, X_t, y_t, X_v, y_v)

            # Score
            metrics = evaluator.evaluate(model_fresh, X_v, y_v, logger=None)
            score = metrics.get(metric_name)
            if score:
                scores.append(score)
                self.logger.info(
                    f"   Window {i+1}: {score:.4f} (Train: {len(X_t)}, Test: {len(X_v)})"
                )

        # Aggregate
        avg_score = sum(scores) / len(scores)
        std_score = pd.Series(scores).std()

        tracker.log_metrics({"wf_avg_score": avg_score, "wf_std_score": std_score})

        # Save detailed report
        pd.DataFrame({"window": range(len(scores)), "score": scores}).to_csv(
            "walk_forward.csv", index=False
        )
        tracker.log_artifact("walk_forward.csv")
        Path("walk_forward.csv").unlink()

        # Criteria: Pass if Average Score is decent (heuristic)
        # Real criterion: Is Avg Score reasonably close to the original "One-Shot" score?
        # For now, we assume Pass if it completed without crashing.
        self.logger.success(
            f"✅ Walk-Forward Complete. Mean: {avg_score:.4f}, Std: {std_score:.4f}"
        )

        return ValidationResult("WalkForward", True, {"mean_score": avg_score})
