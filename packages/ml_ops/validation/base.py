from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
from packages.ml_ops.artifacts import TrainingArtifacts
from packages.ml_ops.tracker import ExperimentTracker


@dataclass
class ValidationResult:
    """Standardized output for any validation step."""

    name: str
    passed: bool
    details: Dict[str, Any]


class BaseValidator(ABC):
    """Abstract contract for all validation plugins."""

    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    def validate(
        self, artifacts: TrainingArtifacts, tracker: ExperimentTracker
    ) -> ValidationResult:
        """
        Takes trained artifacts, runs a validation test, logs results to the tracker,
        and returns a pass/fail result.
        """
        pass
