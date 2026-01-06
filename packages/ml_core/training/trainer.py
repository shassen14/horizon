# packages/ml_core/training/trainer.py

from packages.ml_core.common.artifacts import TrainingArtifacts
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.modeling.pipeline import HorizonPipeline
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
import polars as pl


class HorizonTrainer:
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.factory = factory
        self.logger = logger

    async def train(self, tracker) -> TrainingArtifacts:
        """
        Executes the standard training loop: Load -> Pipeline -> Split -> Fit -> Evaluate.
        """
        # 1. Load Data
        # The factory uses the Blueprint to decide WHICH builder (Alpha vs Regime) to use.
        builder = self.factory.create_dataset_builder(self.blueprint.data)
        raw_df = builder.get_data()
        cache_key = builder._generate_cache_key()

        if raw_df.is_empty():
            raise ValueError("Dataset is empty. Check DB or Builder logic.")

        # Log Metadata
        tracker.log_dataset(raw_df, "training_data", cache_key)

        # 2. Setup Pipeline
        processors = []
        if getattr(self.blueprint.data, "generate_lags", False):
            processors.append(TemporalFeatureProcessor())

        pipeline = HorizonPipeline(
            model=self.factory.create_model(self.blueprint.model),
            feature_prefixes=self.blueprint.data.feature_prefix_groups,
            target=self.blueprint.data.target_column,
            processors=processors,
            exclude_patterns=self.blueprint.data.feature_exclude_patterns,
        )

        # 3. Preprocess & Split
        self.logger.info("Preprocessing Data...")
        processed_df = pipeline.preprocess(raw_df)
        X, y = pipeline.get_X_y(processed_df)

        split_idx = int(len(X) * self.blueprint.training.time_split_ratio)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 4. Fit
        self.logger.info(f"Training on {len(X_train)} samples...")

        # Log Data Profile before training
        tracker.log_data_profile(pl.from_pandas(X_train))

        strategy = self.factory.create_strategy(self.blueprint.training)
        trained_model = strategy.train(pipeline.model, X_train, y_train, X_val, y_val)
        pipeline.model = trained_model

        # 5. Evaluate Baseline
        evaluator = self.factory.create_evaluator(self.blueprint.training)
        metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
        tracker.log_metrics(metrics)

        return TrainingArtifacts(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metrics=metrics,
            feature_names=list(X.columns),
            cache_key=cache_key,
        )
