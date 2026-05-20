"""Runs every validator in order, collects results, reports blocking failures."""

from training.validation.ablation import AblationValidator
from training.validation.base import ValidationResult
from training.validation.generalization import GeneralizationValidator
from training.validation.monte_carlo import MonteCarloValidator
from training.validation.stability import StabilityValidator
from training.validation.walk_forward import WalkForwardValidator

VALIDATORS = [
    GeneralizationValidator(),
    StabilityValidator(),
    AblationValidator(),
    MonteCarloValidator(),
    WalkForwardValidator(),
]


def run_all(model, data) -> list[ValidationResult]:
    return [v.run(model, data) for v in VALIDATORS]
