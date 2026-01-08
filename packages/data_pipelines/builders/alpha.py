# packages/data_pipelines/builders/alpha.py

import polars as pl
from typing import Optional
from .base import AbstractDatasetBuilder
from packages.data_pipelines.labeling.base import AbstractLabeler


class AlphaDatasetBuilder(AbstractDatasetBuilder):
    def __init__(
        self, settings, logger, config, regime_labeler: Optional[AbstractLabeler] = None
    ):
        super().__init__(settings, logger, config)
        self.regime_labeler = regime_labeler

    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info("Building dataset for Alpha Ranking...")

        # 1. Fetch Base Data (All stocks, all features)
        # Note: We assume 'db_url' property is available from Base class
        query = f"""
            SELECT 
                fd.*, 
                mdd.close as close_price,
                a.symbol
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{self.config.start_date}' AND fd.time <= '{self.config.end_date}'
            ORDER BY fd.time ASC
        """

        try:
            df = pl.read_database_uri(query, self.db_url)

            #  Price > $5
            df = df.filter(pl.col("close_price") >= 5.0)

            # Not a massive crash (Down < 75%)
            # Ensure column exists first
            if "high_252_pct" in df.columns:
                df = df.filter(pl.col("high_252_pct") > -0.75)

            # Liquidity
            if "volume_adv_20" in df.columns:
                df = df.filter(
                    (pl.col("close_price") * pl.col("volume_adv_20")) > 2_000_000
                )
        except Exception as e:
            self.logger.error(f"DB Error: {e}")
            return pl.DataFrame()

        if df.is_empty():
            return df

        # 2. Apply Regime Logic (Specific to Alpha Config)
        # We access specific fields safely because we know self.config is AlphaDataConfig
        if self.regime_labeler:
            self.logger.info("Applying regime labels...")
            df = self.regime_labeler.label(df)

            # Now filter based on the new 'regime' column
            if self.config.filter_regime is not None and "regime" in df.columns:
                df = df.filter(pl.col("regime") == self.config.filter_regime)

        # 3. Calculate Target
        # We look forward N days. We must group by asset_id to avoid shifting data between stocks.
        horizon = self.config.target_horizon_days

        df = df.with_columns(
            ((pl.col("close").shift(-horizon) / pl.col("close")) - 1)
            .over("asset_id")
            .alias(self.config.target_column)
        )

        return df
