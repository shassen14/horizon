"""Ablation check (informational): warns if one feature dominates model impact."""

from training.validation.base import BaseValidator, ValidationResult


class AblationValidator(BaseValidator):
    name = "ablation"
    blocking = False

    def run(self, model, data) -> ValidationResult:
        raise NotImplementedError("Phase 6.5")
