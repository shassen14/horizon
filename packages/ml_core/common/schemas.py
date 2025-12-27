from pydantic import BaseModel
from typing import List, Dict, Any


class DataConfig(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    target_horizon_days: int | None = None
    feature_prefix_groups: List[str]

    # Flags
    generate_lags: bool = False
    filter_regime: int | None = None

    dataset_builder: str = "AlphaDatasetBuilder"


class ModelConfig(BaseModel):
    type: str  # e.g., "lightgbm.LGBMRegressor"
    objective: str
    params: Dict[str, Any]


class TrainingConfig(BaseModel):
    strategy: str
    evaluator: str
    time_split_ratio: float
    early_stopping_rounds: int
    eval_metric: str

    # PyTorch specifics (Optional)
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    optimizer: str | None = None
    loss_function: str | None = None


class ModelBlueprint(BaseModel):
    model_name: str
    description: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
