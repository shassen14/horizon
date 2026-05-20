"""Walk-forward check: median fold score clears threshold, with embargo = horizon."""

from training.validation.base import BaseValidator, ValidationResult


class WalkForwardValidator(BaseValidator):
    name = "walk_forward"
    blocking = True

    def run(self, model, data) -> ValidationResult:
        raise NotImplementedError("Phase 6.5")
