import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseValidator, ValidationResult
from packages.contracts.blueprints import ValidationConfig


class StabilityValidator(BaseValidator):
    def __init__(self, logger, config: ValidationConfig):
        super().__init__(logger)
        self.config = config

    def validate(self, artifacts, tracker) -> ValidationResult:
        self.logger.info(
            f"--- Running Stability Validation (Threshold: {self.config.stability_threshold:.1%}) ---"
        )

        model = artifacts.pipeline.model
        X_val = artifacts.X_val
        base_preds = model.predict(X_val)
        X_std = X_val.std()

        results = []
        noise_levels = self.config.stability_noise_levels

        for noise_pct in noise_levels:
            noise = np.random.normal(0, X_std * noise_pct, X_val.shape)
            noisy_preds = model.predict(X_val + noise)

            flips = np.sum(base_preds != noisy_preds)
            rate = flips / len(base_preds) if len(base_preds) > 0 else 0

            results.append({"noise_pct": noise_pct, "flip_rate": rate})
            tracker.log_metrics({f"flip_rate_{int(noise_pct*100)}pct": rate})

        pd.DataFrame(results).to_csv("stability.csv", index=False)
        tracker.log_artifact("stability.csv")
        Path("stability.csv").unlink()

        flip_rate_low_noise = results[0]["flip_rate"]
        passed = flip_rate_low_noise < self.config.stability_threshold

        return ValidationResult(
            name="Stability",
            passed=passed,
            details={"flip_rate_at_1pct_noise": flip_rate_low_noise},
        )
