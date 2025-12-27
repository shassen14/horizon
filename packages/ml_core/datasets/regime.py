# packages/ml_core/datasets/regime.py

import polars as pl
from .base import AbstractDatasetBuilder


class RegimeDatasetBuilder(AbstractDatasetBuilder):
    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info("Building dataset for Regime Classification...")

        start_date = self.config.start_date
        end_date = self.config.end_date

        # 1. Base Query (Same as before)
        query = f"""
            SELECT 
                fd.time, fd.asset_id, fd.sma_50, fd.atr_14_pct, 
                mdd.close as close_price, a.symbol
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{start_date}' AND fd.time <= '{end_date}'
        """

        try:
            full_df = pl.read_database_uri(query, self.db_url)
        except Exception as e:
            self.logger.error(f"DB Error: {e}")
            return pl.DataFrame()

        if full_df.is_empty():
            return pl.DataFrame()

        # 2. Logic specific to Regime Modeling
        # Aggregate Breadth
        market_df = (
            full_df.group_by("time")
            .agg(
                [
                    (pl.col("close_price") > pl.col("sma_50")).sum().cast(pl.Float64)
                    / pl.count(),
                    pl.col("atr_14_pct").mean().alias("vol_market_avg_atr_pct"),
                ]
            )
            .sort("time")
        )

        # Get SPY
        spy_df = (
            full_df.filter(pl.col("symbol") == "SPY")
            .select(["time", "close_price"])
            .rename({"close_price": "spy_close"})
        )

        final_df = market_df.join(spy_df, on="time", how="left")

        # 3. Calculate Target: Is SPY up in 3 months?
        final_df = final_df.with_columns(
            ((pl.col("spy_close").shift(-63) / pl.col("spy_close")) - 1).alias(
                "spy_fwd_ret"
            )
        )

        # Classification Target (1 = Bull, 0 = Bear)
        final_df = final_df.with_columns(
            (pl.col("spy_fwd_ret") > 0.0).cast(pl.Int32).alias("target_regime_bull")
        ).drop(["spy_fwd_ret", "spy_close"])

        return final_df.drop_nulls()
