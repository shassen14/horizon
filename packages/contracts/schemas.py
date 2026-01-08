from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, Union
from .vocabulary.general import OutputType


class BaseOutputSchema(BaseModel):
    """The base contract for all model metadata."""

    description: str


class ClassificationSchema(BaseOutputSchema):
    output_type: Literal[OutputType.CLASSIFICATION]
    classes: List[str]  # e.g., ["BULL", "CHOP", "BEAR"]


class RankingSchema(BaseOutputSchema):
    output_type: Literal[OutputType.RANKING]
    min_value: float = 0.0
    max_value: float = 1.0


class ForecastSchema(BaseOutputSchema):
    output_type: Literal[OutputType.FORECAST]
    unit: str  # e.g., "pct_return", "price_usd"
    horizon_days: int


class DistributionSchema(BaseOutputSchema):
    output_type: Literal[OutputType.DISTRIBUTION]
    metric: str  # e.g., "annualized_vol", "var_95_pct", "expected_shortfall_pct"
    horizon_days: int


# Union type for strict validation
ModelOutputSchema = Annotated[
    Union[ClassificationSchema, RankingSchema, ForecastSchema, DistributionSchema],
    Field(discriminator="output_type"),
]
