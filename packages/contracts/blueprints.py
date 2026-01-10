# packages/ml_core/common/schemas.py

from pydantic import BaseModel, Field
from typing import Annotated, List, Dict, Any, Literal, Union

from .vocabulary.columns import RegimeCol, AlphaCol


# Abstract Base
class BaseDataConfig(BaseModel):
    # Common fields for ALL pipelines
    dataset_builder: str  # Still needed for the Factory registry
    target_column: str
    start_date: str | None = None
    end_date: str | None = None
    feature_prefix_groups: List[str] = []
    feature_exclude_patterns: List[str] = []

    # Caching
    use_cache: bool = True
    force_refresh: bool = False
    cache_tag: str | None = None


class LabelingConfig(BaseModel):
    n_clusters: int = 3
    smoothing_window: int = 5
    forward_cols: List[str] = [RegimeCol.FWD_RET, RegimeCol.FWD_VOL]


# Alpha Implementation
class AlphaDataConfig(BaseDataConfig):
    kind: Literal["alpha"] = "alpha"
    target_horizon_days: int = 63
    generate_lags: bool = True
    regime_model_name: str | None = None
    filter_regime: Union[int, List[int], None] = None
    target_column: str = AlphaCol.TARGET_RETURN


# Regime Implementation
class RegimeDataConfig(BaseDataConfig):
    kind: Literal["regime"] = "regime"
    target_horizon_days: int = 63
    target_column: str = RegimeCol.TARGET
    labeling: LabelingConfig = LabelingConfig()


# The Union Type
# This tells Pydantic: "Look at the 'kind' field. If it's 'alpha', use AlphaDataConfig."
DataConfigType = Annotated[
    Union[AlphaDataConfig, RegimeDataConfig], Field(discriminator="kind")
]


class ModelConfig(BaseModel):
    type: str  # e.g., "lightgbm.LGBMRegressor"
    objective: str
    params: Dict[str, Any]
    dependencies: List[str] = []


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


class ValidationConfig(BaseModel):
    # Stability Checks
    stability_noise_levels: List[float] = [0.01, 0.05, 0.10]
    stability_threshold: float = 0.15  # Max flip rate allowed at lowest noise level

    # Ablation Checks
    ablation_top_n: int = 3  # How many top features to log in summary

    #  Permutation (Monte Carlo)
    monte_carlo_enabled: bool = True
    monte_carlo_simulations: int = 50  # Start small (e.g. 50-100)
    monte_carlo_p_value_threshold: float = 0.05  # 5% significance level

    # Walk Forward
    walk_forward_enabled: bool = True
    walk_forward_windows: int = 5  # Number of expanding windows
    walk_forward_min_train_size: float = 0.5  # Initial window size as % of data

    # Stage 4: WF + Permutation (The "Boss Level")
    # Warning: This is n_windows * n_permutations operations!
    wf_permutation_enabled: bool = False  # Default off to save time


class BacktestConfig(BaseModel):
    enabled: bool = False

    # Explicitly define which Backtester Engine to use
    # "alpha"  -> Uses AlphaBacktester (Ranking/Portfolio logic)
    # "regime" -> Uses RegimeBacktester (Market Timing logic)
    type: Literal["alpha", "regime"] = "alpha"

    # Strategy Class (Only used by AlphaBacktester)
    # e.g. "TopQuintileLongStrategy", "LongShortStrategy"
    strategy_class: str = "TopQuintileLongStrategy"

    # Financial Settings
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0


class ModelBlueprint(BaseModel):
    model_name: str
    description: str
    data: DataConfigType
    model: ModelConfig
    training: TrainingConfig
    validation: ValidationConfig = ValidationConfig()
    backtest: BacktestConfig = BacktestConfig()
