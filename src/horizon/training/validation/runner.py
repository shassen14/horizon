"""Runs every validator in order, collects results, reports blocking failures."""

from horizon.training.validation.ablation import AblationValidator
from horizon.training.validation.base import ValidationResult
from horizon.training.validation.generalization import GeneralizationValidator
from horizon.training.validation.monte_carlo import MonteCarloValidator
from horizon.training.validation.stability import StabilityValidator
from horizon.training.validation.walk_forward import WalkForwardValidator

VALIDATORS = [
    GeneralizationValidator(),
    StabilityValidator(),
    AblationValidator(),
    MonteCarloValidator(),
    WalkForwardValidator(),
]


def run_all(model, data) -> list[ValidationResult]:
    return [v.run(model, data) for v in VALIDATORS]
