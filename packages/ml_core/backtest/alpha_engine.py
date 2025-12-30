import polars as pl
import numpy as np
from typing import Dict, Any
from .base import AbstractBacktester
from .strategies import BacktestStrategy


class AlphaBacktester(AbstractBacktester):
    def __init__(self, strategy: BacktestStrategy, transaction_cost_bps: float):
        self.strategy = strategy
        # Convert BPS to decimal (10 bps = 0.0010)
        self.cost_penalty = transaction_cost_bps / 10000.0

    def run(self, df: pl.DataFrame, target_col: str, pred_col: str) -> Dict[str, Any]:
        """
        Executes the backtest.
        df must contain: [time, asset_id, target_col, pred_col]
        """
        # 1. Ask Strategy for Weights
        weights_df = self.strategy.calculate_weights(df, pred_col)

        # 2. Join Weights with Actual Returns
        # We perform a left join on the original data to keep returns aligned
        df = df.join(weights_df, on=["time", "asset_id"], how="left").fill_null(0.0)

        # 3. Calculate Portfolio Gross Return
        # Sum(Weight * Asset_Return) for each day
        portfolio_df = (
            df.with_columns(
                (pl.col("position_weight") * pl.col(target_col)).alias(
                    "weighted_return"
                )
            )
            .group_by("time")
            .agg(
                [
                    pl.col("weighted_return").sum().alias("gross_return"),
                    # Calculate daily turnover proxy: Sum(|weight|)
                    # Note: True turnover requires T vs T-1 diff, but this is a good
                    # approximation for "Total Exposure Traded" in a rebalancing system.
                    pl.col("position_weight").abs().sum().alias("exposure"),
                ]
            )
            .sort("time")
        )

        # 4. Apply Transaction Costs
        # Assuming we rebalance the full exposure daily (conservative estimate)
        portfolio_df = portfolio_df.with_columns(
            (pl.col("gross_return") - (pl.col("exposure") * self.cost_penalty)).alias(
                "net_return"
            )
        )

        # 5. Calculate Financial Metrics
        returns = portfolio_df["net_return"].to_numpy()

        # Cumulative Return
        cum_ret = np.cumprod(1 + returns)
        total_return = cum_ret[-1] - 1 if len(cum_ret) > 0 else 0.0

        # Annualized Statistics (assuming daily data)
        mean_daily = np.mean(returns)
        std_daily = np.std(returns)

        # Sharpe Ratio (Risk Free Rate assumed 0 for relative ranking comparison)
        sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0.0

        # Max Drawdown
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        return {
            "strategy": self.strategy.__class__.__name__,
            "total_return": float(total_return),
            "annualized_sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "volatility": float(std_daily * np.sqrt(252)),
            "win_rate": float(np.mean(returns > 0)),
        }
