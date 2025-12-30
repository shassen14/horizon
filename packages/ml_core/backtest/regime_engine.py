import polars as pl
import numpy as np
from typing import Dict, Any
from .base import AbstractBacktester


class RegimeBacktester(AbstractBacktester):
    """
    Simulates a Market Timing strategy.
    Switches between a Risk Asset (e.g., SPY) and a Risk-Free Asset (Cash)
    based on the model's classification.
    """

    def __init__(
        self, risk_free_rate_annual: float = 0.04, transaction_cost_bps: float = 5.0
    ):
        self.daily_rfr = risk_free_rate_annual / 252.0
        self.cost_multiplier = transaction_cost_bps / 10000.0

    def run(self, df: pl.DataFrame, target_col: str, pred_col: str) -> Dict[str, Any]:
        """
        df columns expected: [time, target_col (SPY Return), pred_col (0 or 1)]
        """
        # Ensure sorted by time
        df = df.sort("time")

        # 1. Determine Position
        # If Prediction is 1 (Bull), we hold the asset.
        # If Prediction is 0 (Bear), we hold Cash (Risk Free Rate).
        # We shift prediction by 1 because today's signal executes at tomorrow's open/close
        # (Avoiding look-ahead bias is critical here)
        df = df.with_columns(pl.col(pred_col).shift(1).fill_null(1).alias("signal"))

        # 2. Calculate Strategy Returns
        # Strategy = (Signal * Asset_Return) + ((1 - Signal) * Cash_Return)
        df = df.with_columns(
            (
                (pl.col("signal") * pl.col(target_col))
                + ((1 - pl.col("signal")) * self.daily_rfr)
            ).alias("strategy_raw_return")
        )

        # 3. Calculate Transaction Costs
        # We pay costs only when the signal CHANGES (0->1 or 1->0)
        df = df.with_columns(
            (pl.col("signal") - pl.col("signal").shift(1).fill_null(0))
            .abs()
            .alias("trades")
        )

        df = df.with_columns(
            (
                pl.col("strategy_raw_return")
                - (pl.col("trades") * self.cost_multiplier)
            ).alias("strategy_net_return")
        )

        # 4. Calculate Metrics (Vectorized Numpy)
        # We perform calculations on the numpy arrays for speed
        strategy_rets = df["strategy_net_return"].to_numpy()
        benchmark_rets = df[target_col].to_numpy()  # Buy & Hold SPY

        # Helper to calc stats
        def calc_stats(returns):
            cum_ret = np.cumprod(1 + returns)
            total_ret = cum_ret[-1] - 1
            mean = np.mean(returns)
            std = np.std(returns)
            sharpe = (mean / std) * np.sqrt(252) if std > 0 else 0

            # Max Drawdown
            running_max = np.maximum.accumulate(cum_ret)
            dd = (cum_ret - running_max) / running_max
            max_dd = np.min(dd)
            return total_ret, sharpe, max_dd

        strat_total, strat_sharpe, strat_dd = calc_stats(strategy_rets)
        bench_total, bench_sharpe, bench_dd = calc_stats(benchmark_rets)

        return {
            "strategy_total_return": strat_total,
            "strategy_sharpe": strat_sharpe,
            "strategy_max_drawdown": strat_dd,
            "benchmark_total_return": bench_total,
            "benchmark_max_drawdown": bench_dd,
            "outperformance": strat_total - bench_total,
        }
