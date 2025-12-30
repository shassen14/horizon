# packages/ml_core/backtest/strategies.py

from abc import ABC, abstractmethod
import polars as pl


class BacktestStrategy(ABC):
    """
    Defines how model predictions translate into portfolio weights.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def calculate_weights(self, df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
        """
        Input: DataFrame with [time, asset_id, pred_col]
        Output: DataFrame with [time, asset_id, position_weight]

        position_weight should sum to <= 1.0 per timestamp (usually).
        """
        pass


# --- Concrete Implementation: Top N% Long Only ---
class TopQuintileLongStrategy(BacktestStrategy):
    def calculate_weights(self, df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
        # 1. Rank daily
        df = df.with_columns(
            pl.col(pred_col)
            .rank(method="ordinal", descending=True)
            .over("time")
            .alias("daily_rank")
        )

        # 2. Count assets
        df = df.with_columns(pl.count("asset_id").over("time").alias("daily_count"))

        # 3. Select Top 20%
        # Weight = 1.0 (Candidate) or 0.0 (Ignored)
        df = df.with_columns(
            (pl.col("daily_rank") <= (pl.col("daily_count") * 0.20))
            .cast(pl.Float64)
            .alias("raw_weight")
        )

        # 4. Normalize (Equal Weighting among the winners)
        df = df.with_columns(
            pl.col("raw_weight").sum().over("time").alias("total_weight_daily")
        )

        return df.select(
            [
                pl.col("time"),
                pl.col("asset_id"),
                pl.when(pl.col("total_weight_daily") > 0)
                .then(pl.col("raw_weight") / pl.col("total_weight_daily"))
                .otherwise(0.0)
                .alias("position_weight"),
            ]
        )


# --- Concrete Implementation: Long/Short (Market Neutral) ---
class LongShortStrategy(BacktestStrategy):
    def calculate_weights(self, df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
        # Example: Long Top 10%, Short Bottom 10%
        df = df.with_columns(
            pl.col(pred_col)
            .rank(method="ordinal", descending=True)
            .over("time")
            .alias("rank"),
            pl.count("asset_id").over("time").alias("count"),
        )

        # Top 10% get +1, Bottom 10% get -1
        cutoff = pl.col("count") * 0.10

        df = df.with_columns(
            pl.when(pl.col("rank") <= cutoff)
            .then(1.0)
            .when(pl.col("rank") > (pl.col("count") - cutoff))
            .then(-1.0)
            .otherwise(0.0)
            .alias("raw_weight")
        )

        # Normalize Longs and Shorts separately to be 50/50?
        # Or just normalize by total absolute weight.
        df = df.with_columns(
            pl.col("raw_weight").abs().sum().over("time").alias("total_abs_weight")
        )

        return df.select(
            [
                pl.col("time"),
                pl.col("asset_id"),
                (pl.col("raw_weight") / pl.col("total_abs_weight")).alias(
                    "position_weight"
                ),
            ]
        )
