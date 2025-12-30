import importlib
from typing import Type, Any, Dict

# Import Base Classes/Interfaces for type hinting
from packages.ml_core.data.builders.base import AbstractDatasetBuilder
from packages.ml_core.evaluation.base import EvaluationStrategy
from packages.ml_core.training.strategies import TrainingStrategy
from packages.quant_lib.config import Settings

# Import Concrete Implementations for Registration
# (We register these so we can refer to them by simple names in YAML)
from packages.ml_core.training.strategies import (
    SklearnTrainingStrategy,
    PyTorchTrainingStrategy,
)
from packages.ml_core.evaluation.regression import RegressionEvaluator
from packages.ml_core.evaluation.classification import ClassificationEvaluator
from packages.ml_core.data.builders.alpha import AlphaDatasetBuilder
from packages.ml_core.data.builders.regime import RegimeDatasetBuilder
from packages.ml_core.backtest.strategies import (
    TopQuintileLongStrategy,
    LongShortStrategy,
)
from packages.ml_core.backtest.alpha_engine import AlphaBacktester
from packages.ml_core.backtest.regime_engine import RegimeBacktester
from packages.ml_core.backtest.base import AbstractBacktester
from packages.ml_core.common.schemas import BacktestConfig

# from packages.ml_core.datasets.risk import RiskDatasetBuilder


class MLComponentFactory:
    """
    Central Factory for instantiating ML components.
    Uses a Registry pattern for internal components and Dynamic Import for external ones.
    """

    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self.logger = logger

        # --- Registry ---
        # Maps string names from YAML -> Python Classes

        self._dataset_registry: Dict[str, Type[AbstractDatasetBuilder]] = {
            "AlphaDatasetBuilder": AlphaDatasetBuilder,
            "RegimeDatasetBuilder": RegimeDatasetBuilder,
        }

        self._strategy_registry: Dict[str, Type[TrainingStrategy]] = {
            "SklearnTrainingStrategy": SklearnTrainingStrategy,
            "PyTorchTrainingStrategy": PyTorchTrainingStrategy,
        }

        self._evaluator_registry: Dict[str, Type[EvaluationStrategy]] = {
            "RegressionEvaluator": RegressionEvaluator,
            "ClassificationEvaluator": ClassificationEvaluator,
        }

        self._backtest_strategy_registry = {
            "TopQuintileLongStrategy": TopQuintileLongStrategy,
            "LongShortStrategy": LongShortStrategy,
        }

    def create_dataset_builder(self, data_config) -> AbstractDatasetBuilder:
        """Creates the Dataset Builder."""
        builder_name = data_config.dataset_builder
        builder_class = self._dataset_registry.get(builder_name)

        if not builder_class:
            raise ValueError(f"Unknown Dataset Builder: {builder_name}")

        # Inject dependencies (Settings, Logger)
        return builder_class(self.settings, self.logger, data_config)

    def create_model(self, model_config) -> Any:
        """
        Instantiates a model object.
        Supports dynamic loading for PyTorch architectures or Sklearn classes.
        """
        # 1. Parse the type string (e.g. "lightgbm.LGBMRegressor")
        try:
            module_str, class_str = model_config.type.rsplit(".", 1)
            module = importlib.import_module(module_str)
            model_class = getattr(module, class_str)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Factory could not import model type: {model_config.type}"
            ) from e

        # 2. Instantiate with parameters
        try:
            return model_class(**model_config.params)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {class_str} with params {model_config.params}: {e}"
            )

    def create_strategy(self, training_config) -> TrainingStrategy:
        """Creates the Training Strategy (fit loop)."""
        strategy_name = training_config.strategy
        strategy_class = self._strategy_registry.get(strategy_name)

        if not strategy_class:
            raise ValueError(
                f"Unknown Training Strategy: {strategy_name}. Available: {list(self._strategy_registry.keys())}"
            )

        return strategy_class(training_config)

    def create_evaluator(self, training_config) -> EvaluationStrategy:
        """Creates the Evaluator."""
        eval_name = training_config.evaluator
        eval_class = self._evaluator_registry.get(eval_name)

        if not eval_class:
            raise ValueError(f"Unknown Evaluator: {eval_name}")

        return eval_class()

    def create_backtester(self, config: BacktestConfig) -> AbstractBacktester | None:
        """
        Creates the appropriate backtester based on explicit configuration.
        """
        if not config.enabled:
            return None

        # 1. Regime Backtesting
        if config.type == "regime":
            self.logger.info("Initializing Regime Backtester...")
            return RegimeBacktester(
                risk_free_rate_annual=0.04,
                transaction_cost_bps=config.transaction_cost_bps,
            )

        # 2. Alpha Backtesting
        elif config.type == "alpha":
            self.logger.info("Initializing Alpha Backtester...")

            strategy_name = config.strategy_class
            StrategyClass = self._backtest_strategy_registry.get(strategy_name)

            if not StrategyClass:
                self.logger.warning(
                    f"Unknown backtest strategy '{strategy_name}'. Skipping."
                )
                return None

            # Instantiate the Strategy (e.g. TopQuintileLong)
            strategy = StrategyClass(config)

            # Instantiate the Engine
            return AlphaBacktester(strategy, config.transaction_cost_bps)

        else:
            self.logger.error(f"Unknown backtest type: {config.backtest.type}")
            return None
