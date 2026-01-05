# packages/ml_core/training/main.py

import asyncio
import yaml
import sys
from pathlib import Path

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory

# Import the specialized trainers
from packages.ml_core.training.trainers.alpha import AlphaTrainer
from packages.ml_core.training.trainers.regime import RegimeTrainer


async def main(config_path: Path):
    # 1. Load & Validate Blueprint
    # This step ensures the YAML matches our strict Pydantic schemas
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        blueprint = ModelBlueprint.model_validate(config_dict)
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)

    # 2. Setup Logging
    # We name the logger after the model so log files are distinct (e.g. "alpha_bull_v1.json.log")
    log_manager = LogManager(
        service_name=blueprint.model_name, debug=settings.system.debug
    )
    logger = log_manager.get_logger("main")

    logger.info(f"Initializing pipeline from config: {config_path.name}")

    # 3. Setup Factory
    # The factory encapsulates the creation of Strategies, Evaluators, and Backtesters
    factory = MLComponentFactory(settings, logger)

    # 4. Dispatch to Correct Trainer
    # We use the polymorphic 'kind' field from the Data Config to decide.
    data_kind = blueprint.data.kind

    trainer = None

    if data_kind == "alpha":
        logger.info("Detected Alpha Ranking pipeline.")
        trainer = AlphaTrainer(blueprint, factory, logger)

    elif data_kind == "regime":
        logger.info("Detected Market Regime pipeline.")
        trainer = RegimeTrainer(blueprint, factory, logger)

    else:
        logger.error(f"No Trainer implementation found for data kind: '{data_kind}'")
        sys.exit(1)

    # 5. Execute
    try:
        await trainer.run()
    except Exception as e:
        logger.exception("Pipeline failed during execution.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python packages/ml_core/training/main.py <path_to_config.yml>")
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)

    asyncio.run(main(config_file))
