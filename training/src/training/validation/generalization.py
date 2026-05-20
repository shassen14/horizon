"""Generalization check: train vs validation score gap must stay below threshold."""

from training.validation.base import BaseValidator, ValidationResult


class GeneralizationValidator(BaseValidator):
    name = "generalization"
    blocking = True

    def run(self, model, data) -> ValidationResult:
        raise NotImplementedError("Phase 6.5")
