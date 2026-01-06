import numpy as np
import pandas as pd
from pathlib import Path
from packages.ml_core.common.artifacts import TrainingArtifacts
from packages.ml_core.common.tracker import ExperimentTracker
from packages.ml_core.validation.base import BaseValidator, ValidationResult
from packages.ml_core.common.schemas import ValidationConfig


class StabilityValidator(BaseValidator):
    def __init__(self, logger, config: ValidationConfig):
        super().__init__(logger)
        self.config = config

    def validate(
        self, artifacts: TrainingArtifacts, tracker: ExperimentTracker
    ) -> ValidationResult:
        self.logger.info(
            f"Running Stability Validation (Threshold: {self.config.stability_threshold:.1%})..."
        )

        model = artifacts.pipeline.model
        X_val = artifacts.X_val
        base_preds = model.predict(X_val)
        X_std = X_val.std()

        results = []
        noise_levels = self.config.stability_noise_levels  # [0.01, 0.05...]

        for noise_pct in noise_levels:
            # Generate Noise
            noise_matrix = np.random.normal(0, X_std * noise_pct, X_val.shape)
            noisy_preds = model.predict(X_val + noise_matrix)

            flips = np.sum(base_preds != noisy_preds)
            rate = flips / len(base_preds)

            results.append({"noise_pct": noise_pct, "flip_rate": rate})
            tracker.log_metrics({f"flip_rate_{int(noise_pct*100)}pct": rate})

        # Save Artifact
        pd.DataFrame(results).to_csv("stability.csv", index=False)
        tracker.log_artifact("stability.csv")
        Path("stability.csv").unlink()

        # Criteria Check
        # We check the lowest noise level against the threshold
        low_noise_result = results[0]
        flip_rate = low_noise_result["flip_rate"]
        passed = flip_rate < self.config.stability_threshold

        return ValidationResult(
            name="Stability",
            passed=passed,
            details={
                "noise_tested": low_noise_result["noise_pct"],
                "flip_rate": flip_rate,
                "threshold": self.config.stability_threshold,
            },
        )
