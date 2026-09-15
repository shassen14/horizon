"""Shared contract for all validators: the result type and the base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ValidationResult:
    name: str
    passed: bool
    blocking: bool
    metrics: dict
    detail: str


class BaseValidator(ABC):
    name: str
    blocking: bool

    @abstractmethod
    def run(self, model, data) -> ValidationResult:
        """Run this check and return a single result."""
