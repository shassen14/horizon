# packages/ml_core/datasets/regime.py

import polars as pl
from .base import AbstractDatasetBuilder


class RegimeDatasetBuilder(AbstractDatasetBuilder):
    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info("Building dataset for Regime Classification...")

        # 1. Load Data
        query = f"""
            SELECT 
                fd.time, 
                fd.asset_id, 
                fd.sma_50, 
                fd.atr_14_pct,
                fd.rsi_14,
                fd.zscore_20,
                mdd.close as close_price, 
                a.symbol
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{self.config.start_date}' AND fd.time <= '{self.config.end_date}'
        """

        df = pl.read_database_uri(query, self.db_url)
        if df.is_empty():
            return df

        # 2. Calculate Market Breadth (Prefix: breadth_)
        # Logic: (Close > SMA50)
        market_df = (
            df.group_by("time")
            .agg(
                [
                    (
                        (pl.col("close_price") > pl.col("sma_50"))
                        .sum()
                        .cast(pl.Float64)
                        / pl.count()
                    ).alias("breadth_sma50_pct"),
                    # Prefix: vol_
                    pl.col("atr_14_pct").mean().alias("vol_market_avg_atr_pct"),
                ]
            )
            .sort("time")
        )

        # 3. Get SPY Features (Prefix: spy_)
        # We need to make sure SPY actually HAS data in the DB
        spy_df = (
            df.filter(pl.col("symbol") == "SPY")
            .select(["time", "rsi_14", "zscore_20", "close_price"])
            .rename(
                {
                    "rsi_14": "spy_rsi_14",
                    "zscore_20": "spy_zscore_20",
                    "close_price": "spy_close_price",  # Needed for target, not feature
                }
            )
        )

        if spy_df.is_empty():
            self.logger.error("SPY data not found! Regime model cannot be built.")
            return pl.DataFrame()

        # 4. Join
        final_df = market_df.join(spy_df, on="time", how="left")

        # 5. Calculate Target
        final_df = final_df.with_columns(
            (
                (pl.col("spy_close_price").shift(-63) / pl.col("spy_close_price")) - 1
            ).alias("target_return")
        )

        final_df = final_df.with_columns(
            (pl.col("target_return") > 0.0).cast(pl.Int32).alias("target_regime_bull")
        )

        # 6. Cleanup
        # Drop columns used for calculation but not for training
        # We KEEP the ones starting with breadth_, vol_, spy_
        final_df = final_df.drop(["spy_close_price", "target_return"])

        return final_df.drop_nulls()
