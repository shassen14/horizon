from pydantic import BaseModel
from typing import List, Dict, Any


class DataConfig(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    target_horizon_days: int | None = None
    feature_prefix_groups: List[str]
    dataset_builder: str = "AlphaDatasetBuilder"

    # If set, looks for packages/ml_core/models/{name}.pkl
    regime_model_name: str | None = None

    # Flags
    generate_lags: bool = False
    filter_regime: int | None = None

    # Caching Control ---
    use_cache: bool = True  # Default to True for speed
    force_refresh: bool = False  # Set True if you changed DB data or Logic
    cache_tag: str | None = (
        None  # Optional: Manually name the cache file (e.g. "baseline_v1")
    )


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
