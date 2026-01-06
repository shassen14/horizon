from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.common.tracker import ExperimentTracker
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.training.trainer import HorizonTrainer
from packages.ml_core.validation.stability import StabilityValidator
from packages.ml_core.validation.ablation import AblationValidator
from packages.ml_core.training.registry import ModelRegistrar


class CertificationWorkflow:
    def __init__(self, blueprint: ModelBlueprint, factory: MLComponentFactory, logger):
        self.blueprint = blueprint
        self.factory = factory
        self.logger = logger

        # 1. Define the Trainer
        self.trainer = HorizonTrainer(blueprint, factory, logger)

        # 2. Validators
        # Dependencies are injected here.
        self.validators = [
            # Pass the Validation Config (thresholds)
            StabilityValidator(logger, blueprint.validation),
            # Pass Factory/Config for Evaluator creation
            AblationValidator(logger, factory, blueprint.training),
        ]

        self.registrar = ModelRegistrar(logger)

    async def run(self, tracker: ExperimentTracker):
        # A. Execute Training
        self.logger.info(">>> STEP 1: TRAINING")
        artifacts = await self.trainer.train(tracker)

        # B. Log Environment & Profile
        tracker.log_environment(self.blueprint.model.dependencies)

        # C. Run Validation Loop
        self.logger.info(f">>> STEP 2: VALIDATION ({len(self.validators)} Checks)")
        all_passed = True

        for validator in self.validators:
            result = validator.validate(artifacts, tracker)

            if result.passed:
                self.logger.success(f"✅ {result.name} Passed")
            else:
                self.logger.error(f"❌ {result.name} Failed: {result.details}")
                all_passed = False
                # Optional: break here if we want "Fail Fast"

        if not all_passed:
            self.logger.error("⛔ Certification Failed. Model will NOT be registered.")
            tracker.set_tags({"status": "REJECTED"})
            return

        # D. Registration
        self.logger.info(">>> STEP 3: REGISTRATION")
        version = self.registrar.register(self.blueprint, artifacts, tracker)
        tracker.set_tags({"status": "CERTIFIED"})
        self.logger.success(f"🎉 Workflow Complete. Model Version: {version}")
