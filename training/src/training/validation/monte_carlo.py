"""Monte Carlo check: real score must beat date-shuffled permutations (p < 0.05)."""

from training.validation.base import BaseValidator, ValidationResult


class MonteCarloValidator(BaseValidator):
    name = "monte_carlo"
    blocking = True

    def run(self, model, data) -> ValidationResult:
        raise NotImplementedError("Phase 6.5")
