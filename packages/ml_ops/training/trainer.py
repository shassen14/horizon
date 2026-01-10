from packages.contracts.blueprints import ModelBlueprint
from packages.ml_ops.artifacts import TrainingArtifacts
from packages.data_pipelines.processors.temporal import TemporalFeatureProcessor
from packages.ml_ops.modeling.pipeline import HorizonPipeline
from packages.ml_ops.tracker import ExperimentTracker
from .factory import MLComponentFactory
import polars as pl


class HorizonTrainer:
    """
    Orchestrates the model training process based on a blueprint.
    """

    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.factory = factory
        self.logger = logger

    async def train(self, tracker: ExperimentTracker) -> TrainingArtifacts:
        # 1. Build Dataset
        self.logger.info(
            f"Building dataset using '{self.blueprint.data.dataset_builder}'..."
        )
        builder = self.factory.create_dataset_builder(self.blueprint.data)
        raw_df = builder.get_data()
        cache_key = builder._generate_cache_key()

        if raw_df.is_empty():
            raise ValueError(
                "Dataset is empty. Check database connection or builder logic."
            )

        input_example = raw_df.head(5).to_pandas()

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
        self.logger.info(
            "Preprocessing data and splitting into train/validation sets..."
        )
        processed_df = pipeline.preprocess(raw_df)
        X, y = pipeline.get_X_y(processed_df)

        split_idx = int(len(X) * self.blueprint.training.time_split_ratio)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # The pipeline "freezes" the feature list after the first call to get_X_y.
        # We need to capture this list to pass it to the artifacts.
        final_feature_names = pipeline.trained_features
        if final_feature_names is None:
            # This case should not happen if get_X_y was called, but as a fallback:
            final_feature_names = list(X.columns)

        # 4. Fit Model
        self.logger.info(f"Training model on {len(X_train)} samples...")
        tracker.log_data_profile(pl.from_pandas(X_train))

        strategy = self.factory.create_strategy(self.blueprint.training)
        trained_model = strategy.train(pipeline.model, X_train, y_train, X_val, y_val)
        pipeline.model = trained_model

        # 5. Evaluate Baseline Performance
        evaluator = self.factory.create_evaluator(self.blueprint.training)

        # A. Validation Metrics (The Standard)
        val_metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
        tracker.log_metrics(val_metrics)

        # B. Training Metrics (For Overfitting Check)
        # Pass logger=None to keep console clean, we only need the numbers
        train_metrics = evaluator.evaluate(
            pipeline.model, X_train, y_train, logger=None
        )

        # Prefix keys for logging clarity
        tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        # 6. Package and Return Artifacts
        return TrainingArtifacts(
            pipeline=pipeline,
            raw_df=raw_df,  # Pass raw data for validation tests
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_example=input_example,
            metrics=val_metrics,
            train_metrics=train_metrics,
            feature_names=final_feature_names,
            cache_key=cache_key,
        )
