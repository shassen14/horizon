import argparse
import asyncio
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Internal Imports ---
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.modeling.pipeline import HorizonPipeline
from packages.ml_core.data.processors.temporal import TemporalFeatureProcessor
from packages.ml_core.common.tracker import ExperimentTracker


class StabilityEngine:
    def __init__(self, config_path: Path):
        self.config_path = config_path

        # Load Config
        with open(config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)
        self.blueprint = ModelBlueprint.model_validate(self.raw_config)

        # Setup Logging
        self.log_manager = LogManager(
            service_name=f"stability-{self.blueprint.model_name}", debug=True
        )
        self.logger = self.log_manager.get_logger("stability")

        # Setup Factory
        self.factory = MLComponentFactory(settings, self.logger)

    async def run(self):
        run_name = f"stability_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Start MLflow Context
        with ExperimentTracker(self.blueprint.model_name, run_name) as tracker:
            self.logger.info(
                f"🚀 Starting Stability Stress Test. Run ID: {tracker.get_run_id()}"
            )

            tracker.set_tags(
                {
                    "run_type": "stability_test",
                    "target_horizon": str(self.blueprint.data.target_horizon_days),
                }
            )

            tracker.log_params(self.blueprint.model_dump(mode="json"))

            # ---------------------------------------------------------
            # 1. Data Loading & Pipeline Construction (Standard)
            # ---------------------------------------------------------
            self.logger.info("Step 1: Loading and Preprocessing Data...")

            builder = self.factory.create_dataset_builder(self.blueprint.data)
            raw_df = builder.get_data()

            if raw_df.is_empty():
                self.logger.error("Dataset is empty. Aborting.")
                sys.exit(1)

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

            # ---------------------------------------------------------
            # 2. Train / Validation Split
            # ---------------------------------------------------------
            split_idx = int(len(X) * self.blueprint.training.time_split_ratio)

            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            self.logger.info(
                f"Step 2: Training Split. Train: {len(X_train)}, Val: {len(X_val)}"
            )

            # ---------------------------------------------------------
            # 3. Train Baseline Model
            # ---------------------------------------------------------
            self.logger.info("Step 3: Training Baseline Model...")
            training_strategy = self.factory.create_strategy(self.blueprint.training)
            trained_model = training_strategy.train(
                pipeline.model, X_train, y_train, X_val, y_val
            )
            pipeline.model = trained_model

            # ---------------------------------------------------------
            # 4. Get Baseline Predictions
            # ---------------------------------------------------------
            self.logger.info("Step 4: Generating Baseline Predictions...")
            base_preds = pipeline.model.predict(X_val)

            # ---------------------------------------------------------
            # 5. Stability Loop (Noise Injection)
            # ---------------------------------------------------------
            # We test at 1%, 5%, 10%, and 20% noise levels relative to feature StdDev
            noise_levels = [0.01, 0.05, 0.10, 0.20]

            self.logger.info(f"Step 5: Injecting Gaussian Noise ({noise_levels})...")

            results = []

            # Calculate Standard Deviation of features once
            # We use this to scale the noise (so 1% noise on 'price' is different than 1% on 'rsi')
            X_std = X_val.std()

            for noise_pct in noise_levels:
                self.logger.info(f"   ... Testing {int(noise_pct*100)}% Noise Level")

                # A. Generate Noise
                # Noise = Normal(0, StdDev * Pct)
                # We use numpy broadcasting to match the shape of X_val
                noise_matrix = np.random.normal(
                    loc=0.0, scale=X_std * noise_pct, size=X_val.shape
                )

                # B. Add to Validation Set
                # We use a DataFrame to preserve column names for the model
                X_noisy = X_val + noise_matrix

                # C. Predict
                noisy_preds = pipeline.model.predict(X_noisy)

                # D. Compare (Flip Rate)
                # How many predictions are different from the baseline?
                flips = np.sum(base_preds != noisy_preds)
                flip_rate = flips / len(base_preds)

                self.logger.info(
                    f"       -> Flip Rate: {flip_rate:.2%} ({flips}/{len(base_preds)})"
                )

                # Log Metric to MLflow
                tracker.log_metrics({f"flip_rate_{int(noise_pct*100)}pct": flip_rate})

                results.append(
                    {
                        "noise_pct": noise_pct,
                        "flip_rate": flip_rate,
                        "total_flips": flips,
                        "total_samples": len(base_preds),
                    }
                )

            # ---------------------------------------------------------
            # 6. Report & Pass/Fail
            # ---------------------------------------------------------
            df_stab = pd.DataFrame(results)

            # Save CSV
            filename = f"stability_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df_stab.to_csv(filename, index=False)
            tracker.log_artifact(filename)
            Path(filename).unlink()  # Cleanup

            # Check Criteria
            # Fail if > 15% of predictions flip at just 1% noise (Extremely Brittle)
            brittle_threshold = 0.15
            low_noise_flip = df_stab.loc[
                df_stab["noise_pct"] == 0.01, "flip_rate"
            ].values[0]

            self.logger.success("------------------------------------------------")
            if low_noise_flip > brittle_threshold:
                self.logger.error(
                    f"❌ Model is BRITTLE. {low_noise_flip:.1%} flips at 1% noise."
                )
                tracker.set_tags({"stability_check": "FAIL"})
            else:
                self.logger.success(
                    f"✅ Model is STABLE. Only {low_noise_flip:.1%} flips at 1% noise."
                )
                tracker.set_tags({"stability_check": "PASS"})

            self.logger.info(
                f"Full Report uploaded to MLflow Run: {tracker.get_run_id()}"
            )
            self.logger.success("------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon Model Stability Tester")
    parser.add_argument("config", type=Path, help="Path to model .yml blueprint")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"File not found: {args.config}")
        sys.exit(1)

    engine = StabilityEngine(args.config)
    asyncio.run(engine.run())
