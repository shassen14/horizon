from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class PayloadKey(StrEnum):
    """Standard KEYS for the JSONB payload in the 'predictions' table."""

    # General
    PREDICTED_VALUE = "value"  # For simple forecasts (return, volatility, etc.)

    # Classification-specific
    PREDICTED_CLASS = "class"  # String name (e.g., "BULL", "CHOP")
    PROBABILITIES = "probabilities"  # Dict[str, float]

    # Ranking-specific
    SCORE = "score"  # The 0-1 normalized score
    RANK = "rank"  # The ordinal rank

    # Distribution-specific
    VAR_95 = "var_95"  # Value at Risk 95%
    ES_95 = "es_95"  # Expected Shortfall 95%
    CONFIDENCE_INTERVAL = "ci"  # List[float] e.g., [lower_bound, upper_bound]
