"""Stability check: prediction flip rate under small input noise must stay low."""

from training.validation.base import BaseValidator, ValidationResult


class StabilityValidator(BaseValidator):
    name = "stability"
    blocking = True

    def run(self, model, data) -> ValidationResult:
        raise NotImplementedError("Phase 6.5")
