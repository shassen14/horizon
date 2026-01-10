from .base import BaseValidator, ValidationResult
from packages.contracts.blueprints import ValidationConfig


class GeneralizationValidator(BaseValidator):
    def __init__(self, logger, config: ValidationConfig):
        super().__init__(logger)
        self.config = config
        # Default strictness: Allow 20% degradation
        self.threshold = 0.05

    def validate(self, artifacts, tracker) -> ValidationResult:
        self.logger.info("--- Running Generalization Check (Overfitting) ---")

        # Determine main metric
        # We assume standard metrics where they are available in both dicts
        metric = (
            "roc_auc"
            if "roc_auc" in artifacts.metrics
            else list(artifacts.metrics.keys())[0]
        )

        train_score = artifacts.train_metrics.get(metric, 0)
        val_score = artifacts.metrics.get(metric, 0)

        # Calculate degradation
        # Handle directionality: ROC/Accuracy (Higher is Better), LogLoss/MSE (Lower is Better)
        higher_is_better = metric in [
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "r2",
        ]

        if higher_is_better:
            # Drop from 0.99 to 0.70 is bad.
            # (0.99 - 0.70) / 0.99 = 29% Drop
            gap = (train_score - val_score) / abs(train_score)
        else:
            # Rise from 0.1 to 0.5 is bad.
            # (0.5 - 0.1) / 0.1 = 400% Worsening
            gap = (val_score - train_score) / abs(train_score)

        self.logger.info(f"Metric: {metric}")
        self.logger.info(f"  Train Score: {train_score:.4f}")
        self.logger.info(f"  Val Score:   {val_score:.4f}")
        self.logger.info(f"  Degradation: {gap:.1%}")

        passed = gap < self.threshold

        if passed:
            self.logger.success("✅ Generalization Check Passed.")
        else:
            self.logger.error("❌ Generalization Check Failed. Model is Overfitting.")

        return ValidationResult(
            "Generalization", passed, {"gap": gap, "metric": metric}
        )
