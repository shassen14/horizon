# apps/api_server/schemas/explain.py
from datetime import datetime
from pydantic import BaseModel
from typing import List


class FeatureDescription(BaseModel):
    name: str
    family: str
    description: str


class ModelDescription(BaseModel):
    model_name: str
    objective: str
    description: str
    key_features: List[str]


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution_score: float  # SHAP value
    interpretation: str


class DecisionExplanation(BaseModel):
    symbol: str
    model_name: str
    final_score: float
    generated_at: datetime
    feature_contributions: List[FeatureContribution]
