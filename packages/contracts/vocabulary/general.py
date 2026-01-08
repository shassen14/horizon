from enum import Enum


class StrEnum(str, Enum):
    """Base class to make enums behave like strings for easy use in Pydantic/JSON."""

    def __str__(self):
        return self.value


class ModelType(StrEnum):
    """The fundamental ROLE of a model in the pipeline."""

    REGIME = "REGIME"  # Answers: "What is the market's 'weather'?"
    ALPHA = "ALPHA"  # Answers: "Which assets will outperform?"
    RISK = "RISK"  # Answers: "How much danger are we in?"
    EXECUTION = "EXECUTION"  # Answers: "What is the market impact of my trade?"


class OutputType(StrEnum):
    """The standard SHAPES of model outputs, used in schemas.py."""

    CLASSIFICATION = "CLASSIFICATION"  # Discrete classes (e.g., Bull, Bear)
    RANKING = "RANKING"  # A continuous score (e.g., 0.0 to 1.0)
    FORECAST = "FORECAST"  # A specific value prediction (e.g., +2.1% return)
    DISTRIBUTION = "DISTRIBUTION"  # A statistical measure (e.g., Volatility, VaR)
