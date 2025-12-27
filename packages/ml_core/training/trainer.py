# packages/ml_core/training/trainer.py

from pathlib import Path

from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.modeling.pipeline import HorizonPipeline


class Trainer:
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.logger = logger
        self.factory = factory

    async def run(self):
        bp = self.blueprint
        self.logger.info(f"--- Starting Training Pipeline for: {bp.model_name} ---")

        # 1. Load RAW Data (Rows)
        try:
            builder = self.factory.create_dataset_builder(bp.data)
            raw_df = builder.get_data()
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            return

        if raw_df.is_empty():
            self.logger.error("Dataset is empty.")
            return

        # 2. Build Processor List
        processors = []

        # Condition A: Lags
        if getattr(bp.data, "generate_lags", False):
            self.logger.info("Adding TemporalFeatureProcessor to pipeline.")
            processors.append(TemporalFeatureProcessor())

        # Create raw model instance
        raw_model = self.factory.create_model(bp.model)

        pipeline = HorizonPipeline(
            model=raw_model,
            features=bp.data.feature_prefix_groups,
            target=bp.data.target_column,
            processors=processors,
        )

        # 3. Preprocess Data (Columns)
        self.logger.info("Pipeline: Preprocessing data (Features/Lags)...")
        processed_df = pipeline.preprocess(raw_df)

        # 4. Get X and y
        X, y = pipeline.get_X_y(processed_df)

        if X.empty or y is None:
            self.logger.error("Feature extraction failed (Empty X or y).")
            return

        self.logger.info(f"Training Data Shape: {X.shape}")

        # 5. Split
        split_idx = int(len(X) * bp.training.time_split_ratio)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 6. Train (Using Strategy)
        strategy = self.factory.create_strategy(bp.training)

        # We train the *internal model* of the pipeline
        trained_internal_model = strategy.train(
            pipeline.model, X_train, y_train, X_val, y_val
        )

        # Update pipeline with trained model
        pipeline.model = trained_internal_model

        # 7. Evaluate
        evaluator = self.factory.create_evaluator(bp.training)
        # Pass the pipeline to evaluate, or just the model?
        # Usually evaluating the model on X_val is sufficient.
        evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)

        # 8. Save PIPELINE
        model_path = (
            Path(__file__).resolve().parents[1] / "models" / f"{bp.model_name}.pkl"
        )
        pipeline.save(model_path)

        self.logger.success(f"Pipeline saved to {model_path}")
