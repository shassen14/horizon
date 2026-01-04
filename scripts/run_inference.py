import asyncio
import sys
from pathlib import Path

from packages.quant_lib.logging import LogManager
from packages.ml_core.inference import InferenceEngine
from packages.quant_lib.config import settings

sys.path.append(str(Path(__file__).resolve().parents[1]))


async def main():
    print("--- Starting Manual Inference Run ---")

    log_manager = LogManager(service_name="run_inference", debug=settings.system.debug)
    logger = log_manager.get_logger("main")

    engine = InferenceEngine(logger)

    # 1. Generate Rankings
    df_ranks = await engine.run_alpha_ranking()

    if df_ranks is None or df_ranks.is_empty():
        print("❌ No rankings were generated.")
        return

    # 2. Display Results
    print("\n--- 🏆 TOP 10 PICKS ---")
    print(df_ranks.head(10))

    # 3. Save to DB
    await engine.save_predictions(df_ranks)

    print("\n✅ Inference complete and results saved to database.")


if __name__ == "__main__":
    asyncio.run(main())
