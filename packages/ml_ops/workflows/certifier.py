from pathlib import Path
import pandas as pd

# --- MLOps Imports ---
from packages.ml_ops.tracker import ExperimentTracker
from packages.ml_ops.training.factory import MLComponentFactory
from packages.ml_ops.training.trainer import HorizonTrainer
from packages.ml_ops.registry_client import RegistryClient
from packages.ml_ops.validation.stability import StabilityValidator
from packages.ml_ops.validation.ablation import AblationValidator
from packages.ml_ops.validation.monte_carlo import MonteCarloValidator
from packages.ml_ops.validation.walk_forward import WalkForwardValidator

# --- Contract Imports ---
from packages.contracts.blueprints import ModelBlueprint


class CertificationWorkflow:
    """
    Orchestrates the end-to-end model certification pipeline.
    This workflow is the single point of entry for producing a production-ready model.
    """

    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.factory = factory
        self.logger = logger

        # 1. Instantiate the Trainer
        self.trainer = HorizonTrainer(blueprint, factory, logger)

        # 2. Instantiate the Validation Suite (The "Checklist")
        self.validators = [
            StabilityValidator(logger, blueprint.validation),
            AblationValidator(logger, factory, blueprint.training),
            MonteCarloValidator(
                logger,
                factory,
                blueprint.data,
                blueprint.training,
                blueprint.validation,
                blueprint.model,
            ),
            # WalkForwardValidator(
            #     logger,
            #     factory,
            #     blueprint.training,
            #     blueprint.validation,
            #     blueprint.model,
            # ),
        ]

        # 3. Instantiate the Registrar
        self.registry_client = RegistryClient(
            factory.settings.mlflow.tracking_uri, logger
        )

    async def run(self, tracker: ExperimentTracker):
        """Executes the certification pipeline: Train -> Validate -> Register."""

        # --- PRE-FLIGHT CHECKS & LOGGING ---
        tracker.log_environment(self.blueprint.model.dependencies)

        # --- PHASE 1: TRAINING ---
        self.logger.info("=" * 20 + " PHASE 1: MODEL TRAINING " + "=" * 20)
        try:
            artifacts = await self.trainer.train(tracker)
            self.logger.success("✅ Training Phase Complete.")
        except Exception as e:
            self.logger.error(f"❌ Training Phase Failed: {e}", exc_info=True)
            tracker.set_tags({"status": "FAILED", "failure_reason": "training"})
            return

        # Log native feature importance right after training
        self._log_native_importance(
            artifacts.pipeline.model, artifacts.X_train, tracker
        )

        # --- PHASE 2: VALIDATION ---
        self.logger.info("=" * 20 + " PHASE 2: MODEL VALIDATION " + "=" * 20)
        all_passed = True

        for validator in self.validators:
            validator_name = validator.__class__.__name__.replace("Validator", "")

            # Check if validator is enabled in the config
            config_name = f"{validator_name.lower()}_enabled"
            if hasattr(self.blueprint.validation, config_name) and not getattr(
                self.blueprint.validation, config_name
            ):
                self.logger.info(
                    f"--- Skipping {validator_name} Validation (disabled in config) ---"
                )
                continue

            try:
                result = validator.validate(artifacts, tracker)
                if result.passed:
                    self.logger.success(f"  ✅ {result.name} Check: PASSED")
                else:
                    self.logger.error(
                        f"  ❌ {result.name} Check: FAILED. Details: {result.details}"
                    )
                    all_passed = False
            except Exception as e:
                self.logger.error(
                    f"  ❌ {validator_name} Check: CRASHED. Error: {e}", exc_info=True
                )
                all_passed = False

        if not all_passed:
            self.logger.error(
                "⛔ Certification Failed. One or more validation checks did not pass."
            )
            tracker.set_tags({"status": "REJECTED", "failure_reason": "validation"})
            return

        self.logger.success("✅ Validation Phase Complete. All checks passed.")

        # --- PHASE 3: REGISTRATION ---
        self.logger.info("=" * 20 + " PHASE 3: MODEL REGISTRATION " + "=" * 20)
        try:
            version = self.registry_client.register_pipeline(
                self.blueprint, artifacts, tracker
            )
            tracker.set_tags({"status": "CERTIFIED"})
            self.logger.success(
                f"🎉 Model Certified & Registered! Name: '{self.blueprint.model_name}', Version: {version}"
            )
        except Exception as e:
            self.logger.error(f"❌ Registration Phase Failed: {e}", exc_info=True)
            tracker.set_tags({"status": "FAILED", "failure_reason": "registration"})

    def _log_native_importance(self, model, X_features: pd.DataFrame, tracker):
        """Logs the model's built-in feature importance attribute if it exists."""
        if hasattr(model, "feature_importances_"):
            self.logger.info("Logging native feature importances...")
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
