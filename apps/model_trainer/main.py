# apps/model_trainer/main.py

import argparse
import asyncio
from pathlib import Path
import yaml
from datetime import datetime

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.common.tracker import ExperimentTracker
from packages.ml_core.workflows.certifier import CertificationWorkflow


async def main(config_path: Path):
    # 1. Setup
    with open(config_path, "r") as f:
        blueprint = ModelBlueprint.model_validate(yaml.safe_load(f))

    log_manager = LogManager(f"certify-{blueprint.model_name}", debug=True)
    logger = log_manager.get_logger("main")
    factory = MLComponentFactory(settings, logger)

    # 2. Initialize Workflow
    workflow = CertificationWorkflow(blueprint, factory, logger)

    # 3. Run within Tracker Context
    run_name = f"certify_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with ExperimentTracker(blueprint.model_name, run_name) as tracker:
        await workflow.run(tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.config))
