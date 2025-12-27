# packages/ml_core/training/trainer.py

from datetime import datetime
from pathlib import Path
import mlflow

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

        # Setup MLflow
        # Set the experiment name (creates it if it doesn't exist)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(bp.model_name)

        with mlflow.start_run(
            run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            # Log Configuration (Flattens the nested Pydantic config)
            # This lets you search: "Show me all runs where learning_rate > 0.05"
            # Log all configuration parameters for reproducibility
            mlflow.log_params(bp.model_dump(mode="json"))

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

            # (Future TODO: Add other processors here, e.g. Scalers)

            # Create raw model instance
            try:
                raw_model = self.factory.create_model(bp.model)
            except Exception as e:
                self.logger.error(f"Model creation failed: {e}")
                return

            # We create the pipeline with our list of processors
            pipeline = HorizonPipeline(
                model=raw_model,
                features=bp.data.feature_prefix_groups,
                target=bp.data.target_column,
                processors=processors,
            )

            # 3. Preprocess Data (Columns)
            self.logger.info("Pipeline: Preprocessing data...")
            # This runs the processors and drops nulls
            processed_df = pipeline.preprocess(raw_df)

            # 4. Get X and y
            # The pipeline selects columns based on 'features' prefixes
            X, y = pipeline.get_X_y(processed_df)

            if X.empty or y is None:
                self.logger.error("Feature extraction failed (Empty X or y).")
                return

            self.logger.info(f"Training Data Shape: {X.shape}")
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param(
                "feature_names", list(X.columns)
            )  # Log exact features used

            # 5. Split
            split_idx = int(len(X) * bp.training.time_split_ratio)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            self.logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

            # 6. Train (Using Strategy)
            strategy = self.factory.create_strategy(bp.training)

            # We train the internal model of the pipeline
            trained_internal_model = strategy.train(
                pipeline.model, X_train, y_train, X_val, y_val
            )

            # Update pipeline with trained model
            pipeline.model = trained_internal_model

            # 7. Evaluate
            evaluator = self.factory.create_evaluator(bp.training)
            # Pass the pipeline to evaluate, or just the model?
            # Usually evaluating the model on X_val is sufficient.
            metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
            mlflow.log_metrics(metrics)

            # 8. Save PIPELINE
            model_path = (
                Path(__file__).resolve().parents[1] / "models" / f"{bp.model_name}.pkl"
            )
            pipeline.save(model_path)

            # Upload to MLflow (Versioning)
            mlflow.log_artifact(str(model_path))

            self.logger.success(f"Pipeline saved to {model_path}")
