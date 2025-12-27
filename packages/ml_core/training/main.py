import asyncio
import yaml
import sys
from pathlib import Path

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.training.trainer import Trainer


async def main(config_path: Path):
    # 1. Load Blueprint
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    blueprint = ModelBlueprint.model_validate(config_dict)

    # 2. Setup Logger
    log_manager = LogManager(
        service_name=blueprint.model_name, debug=settings.system.debug
    )
    logger = log_manager.get_logger("trainer")

    # 3. Setup Factory
    # The factory encapsulates the complexity of object creation
    factory = MLComponentFactory(settings, logger)

    # 4. Inject Dependencies into Trainer
    trainer = Trainer(blueprint=blueprint, factory=factory, logger=logger)

    # 5. Run
    await trainer.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python packages/ml_core/training/main.py <path_to_config.yml>")
        sys.exit(1)

    config_file = Path(sys.argv[1])
    asyncio.run(main(config_file))
