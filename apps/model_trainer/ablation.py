import argparse
import asyncio
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.modeling.pipeline import HorizonPipeline
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
from packages.ml_core.common.tracker import ExperimentTracker


class AblationEngine:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)
        self.blueprint = ModelBlueprint.model_validate(self.raw_config)

        self.log_manager = LogManager(
            f"ablation-{self.blueprint.model_name}", debug=True
        )
        self.logger = self.log_manager.get_logger("ablation")
        self.factory = MLComponentFactory(settings, self.logger)

    async def run(self):
        # Use the Model Name as the Experiment, but append "_diagnostics" or keep same?
        # Let's keep same experiment so we see training vs ablation side-by-side
        run_name = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Start MLflow Context
        with ExperimentTracker(self.blueprint.model_name, run_name) as tracker:
            self.logger.info(
                f"🚀 Starting Feature Ablation. Run ID: {tracker.get_run_id()}"
            )

            # Tag this run so we can filter it easily in UI
            tracker.set_tags(
                {
                    "run_type": "ablation",
                    "target_horizon": str(self.blueprint.data.target_horizon_days),
                }
            )

            # Log the Config used
            tracker.log_params(self.blueprint.model_dump(mode="json"))

            # --- 1. Load & Process (Same as before) ---
            builder = self.factory.create_dataset_builder(self.blueprint.data)
            raw_df = builder.get_data()

            # ... (Load Processor, Create Pipeline, Preprocess, Get X/y) ...
            # [Re-use your existing logic here to get X, y, pipeline]
            # For brevity, assuming you copy-paste the logic from previous `ablation.py` here:
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
            feature_names = list(X.columns)

            # --- 2. Train Baseline ---
            split_idx = int(len(X) * self.blueprint.training.time_split_ratio)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            strategy = self.factory.create_strategy(self.blueprint.training)
            trained_model = strategy.train(
                pipeline.model, X_train, y_train, X_val, y_val
            )
            pipeline.model = trained_model

            # --- 3. Establish Baseline ---
            evaluator = self.factory.create_evaluator(self.blueprint.training)
            base_metrics = evaluator.evaluate(pipeline.model, X_val, y_val, self.logger)

            # Log Baseline Metrics to MLflow
            tracker.log_metrics({f"baseline_{k}": v for k, v in base_metrics.items()})

            primary_metric = self.blueprint.training.eval_metric
            base_score = base_metrics.get(
                primary_metric, list(base_metrics.values())[0]
            )

            # --- 4. Ablation Loop ---
            results = []
            for feat in feature_names:
                X_corrupted = X_val.copy()
                X_corrupted[feat] = np.random.permutation(X_corrupted[feat].values)

                # Evaluate
                # Note: We silence logger here to keep logs clean
                c_metrics = evaluator.evaluate(
                    pipeline.model, X_corrupted, y_val, logger=None
                )
                c_score = c_metrics.get(primary_metric, list(c_metrics.values())[0])

                delta = (
                    c_score - base_score
                )  # Assuming Higher is Better? Check metric logic.
                # For LogLoss, Lower is Better, so delta > 0 means it got worse (Good feature)
                # For Accuracy, Higher is Better, so delta < 0 means it got worse (Good feature)

                # Let's standardize on "Impact Magnitude"
                impact = abs(c_score - base_score)

                results.append(
                    {
                        "feature": feat,
                        "baseline": base_score,
                        "corrupted": c_score,
                        "delta": delta,
                        "impact": impact,
                    }
                )

            # --- 5. Save & Log Artifacts ---
            df_res = pd.DataFrame(results).sort_values("impact", ascending=False)

            # Save CSV locally then upload
            csv_name = "feature_ablation.csv"
            df_res.to_csv(csv_name, index=False)
            tracker.log_artifact(csv_name)

            # Clean up local file
            Path(csv_name).unlink()

            self.logger.success(
                f"Ablation uploaded to MLflow (Run {tracker.get_run_id()})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon Feature Ablation Tool")
    parser.add_argument("config", type=Path, help="Path to model .yml blueprint")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"File not found: {args.config}")
        sys.exit(1)

    engine = AblationEngine(args.config)
    asyncio.run(engine.run())
