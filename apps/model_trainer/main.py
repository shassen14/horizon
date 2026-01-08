import argparse
import asyncio
from pathlib import Path
import yaml
from datetime import datetime

# --- Library Imports ---
# These imports show the clean separation of concerns.
# We import from each of our specialized packages.
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager

from packages.contracts.blueprints import ModelBlueprint, LabelingConfig

from packages.data_pipelines.labeling.engine import LabelingEngine

from packages.ml_ops.training.factory import MLComponentFactory
from packages.ml_ops.workflows.certifier import CertificationWorkflow
from packages.ml_ops.tracker import ExperimentTracker

# --- Main Application Logic ---


async def main():
    # --- 1. Argument Parsing (The Dispatcher) ---
    parser = argparse.ArgumentParser(description="Horizon MLOps Command Center")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # A. `label` command
    parser_label = subparsers.add_parser(
        "label", help="Generate offline regime label artifacts."
    )
    parser_label.add_argument(
        "horizons", type=int, nargs="+", help="Horizons in days (e.g., 21 63)."
    )

    # B. `certify` command
    parser_certify = subparsers.add_parser(
        "certify", help="Run the full training and validation workflow for a model."
    )
    parser_certify.add_argument(
        "config", type=Path, help="Path to the model's .yml blueprint."
    )

    # C. `run-test` command (for quick, isolated validation runs)
    parser_test = subparsers.add_parser(
        "run-test", help="Run a single validation test on a trained model."
    )
    parser_test.add_argument(
        "test_name",
        choices=["stability", "ablation", "monte_carlo"],
        help="Name of the test to run.",
    )
    parser_test.add_argument(
        "config", type=Path, help="Path to the model's .yml blueprint."
    )

    args = parser.parse_args()

    # --- 2. Command Execution ---
    if args.command == "label":
        engine = LabelingEngine()
        for h in args.horizons:
            # Using default LabelingConfig for now, can be extended to take YAML
            engine.run(horizon=h, config=LabelingConfig())

    elif args.command == "certify":
        await run_certification(args.config)

    elif args.command == "run-test":
        await run_single_test(args.config, args.test_name)


async def run_certification(config_path: Path):
    """Orchestrates the main certification workflow."""
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        return

    # 1. Setup
    with open(config_path, "r") as f:
        blueprint = ModelBlueprint.model_validate(yaml.safe_load(f))

    log_manager = LogManager(f"certify-{blueprint.model_name}", debug=True)
    logger = log_manager.get_logger("main")
    factory = MLComponentFactory(settings, logger)

    # 2. Initialize Workflow
    workflow = CertificationWorkflow(blueprint, factory, logger)

    # 3. Run within a single, overarching MLflow Tracker context
    run_name = f"certify_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with ExperimentTracker(blueprint.model_name, run_name) as tracker:
        await workflow.run(tracker)


async def run_single_test(config_path: Path, test_name: str):
    """Runs one specific validation test for rapid analysis."""
    # This is a simplified version of the certifier for brevity.
    # It performs the training step to get the artifacts, then runs only one validator.
    # A more advanced version might load an already trained model.
    print(f"--- Running isolated test: {test_name} ---")

    with open(config_path, "r") as f:
        blueprint = ModelBlueprint.model_validate(yaml.safe_load(f))

    log_manager = LogManager(f"test-{test_name}-{blueprint.model_name}", debug=True)
    logger = log_manager.get_logger("main")
    factory = MLComponentFactory(settings, logger)

    # Import validators here to keep dependencies clean
    from packages.ml_ops.validation.stability import StabilityValidator
    from packages.ml_ops.validation.ablation import AblationValidator
    from packages.ml_ops.validation.monte_carlo import MonteCarloValidator
    from packages.ml_ops.training.trainer import HorizonTrainer

    # Run within a tracker
    run_name = f"test_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with ExperimentTracker(blueprint.model_name, run_name) as tracker:
        tracker.set_tags({"run_type": f"isolated_test_{test_name}"})

        # 1. Train the model to get the necessary artifacts
        logger.info("Training baseline model to generate artifacts for testing...")
        trainer = HorizonTrainer(blueprint, factory, logger)
        artifacts = await trainer.train(tracker)

        # 2. Select and run the requested validator
        validator = None
        if test_name == "stability":
            validator = StabilityValidator(logger, blueprint.validation)
        elif test_name == "ablation":
            validator = AblationValidator(logger, factory, blueprint.training)
        elif test_name == "monte_carlo":
            validator = MonteCarloValidator(
                logger,
                factory,
                blueprint.data,
                blueprint.training,
                blueprint.validation,
                blueprint.model,
            )

        if validator:
            logger.info(f"Executing '{test_name}' validation...")
            result = validator.validate(artifacts, tracker)
            logger.info(
                f"Test complete. Passed: {result.passed}. Details: {result.details}"
            )
        else:
            logger.error(f"Test '{test_name}' not implemented.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n🔥 An unexpected error occurred: {e}")
