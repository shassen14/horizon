# apps/model_trainer/main.py

import asyncio
import sys
import argparse
from pathlib import Path
import yaml

# Import from the Shared Library
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.common.schemas import ModelBlueprint
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.training.trainers.alpha import AlphaTrainer
from packages.ml_core.training.trainers.regime import RegimeTrainer


async def run_training(config_path: Path):
    # 1. Setup Logging
    # We load the config just to get the model name for the log file
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    model_name = raw_config.get("model_name", "unknown_model")

    # Debug = True because we are on Desktop
    log_manager = LogManager(service_name=f"train-{model_name}", debug=True)
    logger = log_manager.get_logger("main")

    logger.info(f"🚀 Starting Trainer CLI using config: {config_path.name}")

    # 2. Validate Config (Strict Schema)
    try:
        blueprint = ModelBlueprint.model_validate(raw_config)
    except Exception as e:
        logger.error(f"❌ Configuration Invalid:\n{e}")
        return

    # 3. Initialize Factory
    factory = MLComponentFactory(settings, logger)

    # 4. Dispatcher Pattern
    # The blueprint tells us WHAT it is (Alpha vs Regime), we pick the handler.
    trainer = None
    data_kind = blueprint.data.kind

    if data_kind == "alpha":
        logger.info("🔵 Mode: Alpha Strategy Training")
        trainer = AlphaTrainer(blueprint, factory, logger)

    elif data_kind == "regime":
        logger.info("🟠 Mode: Market Regime Classification")
        trainer = RegimeTrainer(blueprint, factory, logger)

    else:
        logger.critical(f"❌ No trainer found for kind: {data_kind}")
        return

    # 5. Execute
    try:
        await trainer.run()
        logger.success("✅ Training Session Complete.")
    except Exception:
        logger.exception("🔥 Training Crashed.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon Model Trainer")
    parser.add_argument("config", type=Path, help="Path to model .yml blueprint")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"File not found: {args.config}")
        sys.exit(1)

    # Run Async Loop
    asyncio.run(run_training(args.config))
