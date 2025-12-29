# apps/db_worker/main.py

import asyncio
import argparse
from packages.quant_lib.logging import LogManager

from packages.quant_lib.market_clock import MarketClock
from policies import PolicyManager
from auditor import DataAuditor
from optimization import DatabaseOptimizer


async def main():
    parser = argparse.ArgumentParser(description="Horizon Database Worker")
    parser.add_argument(
        "--mode", choices=["all", "policies", "audit", "optimize"], default="all"
    )
    args = parser.parse_args()

    log_manager = LogManager("db-worker")
    logger = log_manager.get_logger("main")
    clock = MarketClock()

    try:
        # Policies (Set and Forget)
        if args.mode in ["all", "policies"]:
            pm = PolicyManager(logger)
            await pm.apply_all()

        # Audit (Find gaps)
        if args.mode in ["all", "audit"]:
            auditor = DataAuditor(logger, clock)
            await auditor.run_daily_audit()
            # Intraday audit is heavy, maybe run separately or strictly scheduled
            await auditor.run_intraday_audit()

        # Optimize (Cleanup)
        if args.mode in ["all", "optimize"]:
            opt = DatabaseOptimizer(logger)
            await opt.run_maintenance()

    except Exception as e:
        logger.exception("CRITICAL FAILURE in DB Worker")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
