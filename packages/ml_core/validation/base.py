from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

from packages.ml_core.common.artifacts import TrainingArtifacts
from packages.ml_core.common.tracker import ExperimentTracker


@dataclass
class ValidationResult:
    name: str
    passed: bool
    details: Dict[str, Any]


class BaseValidator(ABC):
    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    def validate(
        self, artifacts: TrainingArtifacts, tracker: ExperimentTracker
    ) -> ValidationResult:
        pass
