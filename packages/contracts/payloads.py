from pydantic import BaseModel
from typing import Dict


class ClassificationPayload(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]


class RankingPayload(BaseModel):
    score: float  # The normalized score


class ForecastPayload(BaseModel):
    value: float


class DistributionPayload(BaseModel):
    value: float
