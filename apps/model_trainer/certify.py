# apps/model_trainer/certify.py

import argparse
import asyncio
import sys
import yaml
import mlflow
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime

# --- Internal Imports ---
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.modeling.pipeline import HorizonPipeline, HorizonMLflowWrapper
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
from packages.ml_core.common.tracker import ExperimentTracker

# Analyzers
from packages.ml_core.validation.analyzers import StabilityAnalyzer, AblationAnalyzer


class ModelCertifier:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)
        self.blueprint = ModelBlueprint.model_validate(self.raw_config)

        self.log_manager = LogManager(
            f"certify-{self.blueprint.model_name}", debug=True
        )
        self.logger = self.log_manager.get_logger("certify")
        self.factory = MLComponentFactory(settings, self.logger)

    async def run(self):
        run_name = f"certify_{datetime.now().strftime('%Y%m%d_%H%M')}"

        with ExperimentTracker(self.blueprint.model_name, run_name) as tracker:
            self.logger.info(
                f"🚀 Starting Certification Pipeline. Run ID: {tracker.get_run_id()}"
            )

            # Log Config
            tracker.log_params(self.blueprint.model_dump(mode="json"))
            tracker.set_tags({"run_type": "certification"})

            # [ADDED] 1. Log Environment (requirements.txt)
            tracker.log_environment(self.blueprint.model.dependencies)

            # ------------------------------------------------------------------
            # PHASE 1: TRAIN
            # ------------------------------------------------------------------
            self.logger.info("=== PHASE 1: TRAINING ===")

            # A. Load Data
            builder = self.factory.create_dataset_builder(self.blueprint.data)
            raw_df = builder.get_data()
            if raw_df.is_empty():
                self.logger.error("Dataset Empty!")
                sys.exit(1)

            # [ADDED] 2. Log Dataset Metadata (Hash, Schema)
            # This tags the run with 'dataset_cache_key' so you know exactly which data was used
            cache_key = builder._generate_cache_key()
            tracker.log_dataset(raw_df, "training_data", cache_key)

            # B. Pipeline
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
            processed_df = pipeline.preprocess(raw_df)
            X, y = pipeline.get_X_y(processed_df)

            # C. Split & Train
            split_idx = int(len(X) * self.blueprint.training.time_split_ratio)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            # [ADDED] 3. Log Data Profile (HTML Stats)
            # We convert X_train back to Polars for the profiler
            tracker.log_data_profile(pl.from_pandas(X_train))

            self.logger.info(f"Training on {len(X_train)} samples...")
            strategy = self.factory.create_strategy(self.blueprint.training)
            trained_model = strategy.train(
                pipeline.model, X_train, y_train, X_val, y_val
            )
            pipeline.model = trained_model

            # D. Evaluate Baseline
            evaluator = self.factory.create_evaluator(self.blueprint.training)
            metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)
            tracker.log_metrics(metrics)

            # [ADDED] 4. Log Native Feature Importance (if supported)
            self._log_native_importance(pipeline.model, X_train, tracker)

            # ------------------------------------------------------------------
            # PHASE 2: STABILITY TEST
            # ------------------------------------------------------------------
            self.logger.info("=== PHASE 2: STABILITY CHECK ===")
            stab_analyzer = StabilityAnalyzer(self.logger)
            df_stab = stab_analyzer.run(pipeline.model, X_val)

            for _, row in df_stab.iterrows():
                tracker.log_metrics(
                    {f"flip_rate_{int(row['noise_pct']*100)}pct": row["flip_rate"]}
                )

            df_stab.to_csv("stability.csv", index=False)
            tracker.log_artifact("stability.csv")
            Path("stability.csv").unlink()

            flip_1pct = df_stab.iloc[0]["flip_rate"]
            if flip_1pct > 0.15:
                self.logger.error(
                    f"❌ STABILITY FAIL: {flip_1pct:.1%} flips at 1% noise."
                )
                tracker.set_tags({"status": "REJECTED", "failure_reason": "stability"})
                return
            else:
                self.logger.success(f"✅ Stability Pass ({flip_1pct:.1%})")

            # ------------------------------------------------------------------
            # PHASE 3: ABLATION (FEATURE IMPORTANCE)
            # ------------------------------------------------------------------
            self.logger.info("=== PHASE 3: ABLATION ===")
            abl_analyzer = AblationAnalyzer(
                self.logger, evaluator, self.blueprint.training.eval_metric
            )
            df_abl = abl_analyzer.run(pipeline.model, X_val, y_val, metrics)

            df_abl.to_csv("ablation.csv", index=False)
            tracker.log_artifact("ablation.csv")
            Path("ablation.csv").unlink()
            self.logger.success("✅ Ablation Complete")

            # ------------------------------------------------------------------
            # PHASE 4: REGISTRATION
            # ------------------------------------------------------------------
            self.logger.info("=== PHASE 4: REGISTRATION ===")

            input_example = X_val.head(5)
            # Ensure float types for signature safety
            int_cols = input_example.select_dtypes(include=["int"]).columns
            if len(int_cols) > 0:
                input_example = input_example.copy()
                input_example[int_cols] = input_example[int_cols].astype(float)

            # [ADDED] 5. Generate & Log Model Signature (JSON)
            # This method in tracker logs 'model_signature.json' to artifacts
            prediction_example = pipeline.predict(pl.from_pandas(input_example))
            signature = tracker.log_model_signature(input_example, prediction_example)

            # Log Model
            mlflow_model = HorizonMLflowWrapper(pipeline)
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=mlflow_model,
                registered_model_name=self.blueprint.model_name,
                signature=signature,
                input_example=input_example,
                pip_requirements=self.blueprint.model.dependencies,
            )

            tracker.set_tags({"status": "CERTIFIED"})
            self.logger.success(
                f"🎉 Model Certified & Registered! Version: {model_info.registered_model_version}"
            )

    def _log_native_importance(
        self, model, X_features: pd.DataFrame, tracker: ExperimentTracker
    ):
        """Logs the model's internal feature importance (e.g. Gini importance for RF)"""
        if hasattr(model, "feature_importances_"):
            self.logger.info("Logging Native Feature Importance...")
            imp_df = pd.DataFrame(
                {
                    "feature": list(X_features.columns),
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)

            imp_path = "feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            tracker.log_artifact(imp_path)
            Path(imp_path).unlink()
        elif hasattr(model, "coef_"):
            # For Linear Models
            self.logger.info("Logging Coefficients...")
            imp_df = pd.DataFrame(
                {
                    "feature": list(X_features.columns),
                    "importance": (
                        model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    ),
                }
            ).sort_values(by="importance", key=abs, ascending=False)

            imp_path = "feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            tracker.log_artifact(imp_path)
            Path(imp_path).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon Model Certification")
    parser.add_argument("config", type=Path, help="Path to model .yml blueprint")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"File not found: {args.config}")
        sys.exit(1)

    asyncio.run(ModelCertifier(args.config).run())
