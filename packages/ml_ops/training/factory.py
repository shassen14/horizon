import importlib
from typing import Any
from packages.contracts.blueprints import DataConfigType
from packages.quant_lib.config import Settings
from packages.data_pipelines.builders.base import AbstractDatasetBuilder
from packages.ml_ops.evaluation.base import EvaluationStrategy
from .strategies import (
    TrainingStrategy,
    SklearnTrainingStrategy,
    PyTorchTrainingStrategy,
)

from packages.data_pipelines.builders.alpha import AlphaDatasetBuilder
from packages.data_pipelines.builders.regime import RegimeDatasetBuilder
from packages.ml_ops.evaluation.classification import ClassificationEvaluator
from packages.ml_ops.evaluation.regression import RegressionEvaluator


class MLComponentFactory:
    """
    Central Factory for instantiating ML components based on configuration.
    Uses a Registry pattern to map string names from YAML to Python classes.
    """

    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self.logger = logger

        # --- Component Registries ---
        self._dataset_registry = {
            "AlphaDatasetBuilder": AlphaDatasetBuilder,
            "RegimeDatasetBuilder": RegimeDatasetBuilder,
        }
        self._strategy_registry = {
            "SklearnTrainingStrategy": SklearnTrainingStrategy,
            "PyTorchTrainingStrategy": PyTorchTrainingStrategy,
        }
        self._evaluator_registry = {
            "ClassificationEvaluator": ClassificationEvaluator,
            "RegressionEvaluator": RegressionEvaluator,
        }

    def create_dataset_builder(
        self, data_config: DataConfigType
    ) -> AbstractDatasetBuilder:
        builder_name = data_config.dataset_builder
        builder_class = self._dataset_registry.get(builder_name)
        if not builder_class:
            raise ValueError(f"Unknown Dataset Builder: '{builder_name}'")

        # Case 1: AlphaDatasetBuilder requires the 'regime_labeler'
        if builder_name == "AlphaDatasetBuilder":
            regime_labeler = None
            if data_config.regime_model_name:
                from packages.ml_ops.registry_client import RegistryClient
                from packages.ml_ops.labelers.regime import RegimeModelLabeler

                self.logger.info(
                    "Instantiating RegimeModelLabeler dependency for Alpha builder..."
                )
                registry_client = RegistryClient(
                    self.settings.mlflow.tracking_uri, self.logger
                )
                regime_labeler = RegimeModelLabeler(
                    registry_client, data_config.regime_model_name, logger=self.logger
                )

            # Inject the labeler (or None) into the constructor
            return builder_class(
                self.settings, self.logger, data_config, regime_labeler=regime_labeler
            )

        # Case 2: Default for all other builders (like RegimeDatasetBuilder)
        else:
            # They have a simpler constructor and do not accept 'regime_labeler'
            return builder_class(self.settings, self.logger, data_config)

    def create_model(self, model_config) -> Any:
        try:
            module_str, class_str = model_config.type.rsplit(".", 1)
            module = importlib.import_module(module_str)
            model_class = getattr(module, class_str)
            return model_class(**model_config.params)
        except (ImportError, AttributeError, TypeError) as e:
            raise ValueError(
                f"Factory could not create model '{model_config.type}': {e}"
            ) from e

    def create_strategy(self, training_config) -> TrainingStrategy:
        strategy_name = training_config.strategy
        strategy_class = self._strategy_registry.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown Training Strategy: '{strategy_name}'")
        return strategy_class(training_config)

    def create_evaluator(self, training_config) -> EvaluationStrategy:
        eval_name = training_config.evaluator
        eval_class = self._evaluator_registry.get(eval_name)
        if not eval_class:
            raise ValueError(f"Unknown Evaluator: '{eval_name}'")
        return eval_class()
