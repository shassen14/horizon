import polars as pl
from typing import Optional

# --- Local Imports ---
from .base import AbstractDatasetBuilder
from packages.data_pipelines.labeling.base import AbstractLabeler

# --- Contract Imports ---
from packages.contracts.vocabulary.columns import MarketCol, RegimeCol


class AlphaDatasetBuilder(AbstractDatasetBuilder):
    """
    Builds datasets for Alpha (Stock Ranking) models.
    Fetches stock data AND market context to enable regime filtering.
    """

    def __init__(
        self, settings, logger, config, regime_labeler: Optional[AbstractLabeler] = None
    ):
        super().__init__(settings, logger, config)
        self.regime_labeler = regime_labeler

    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info("Building dataset for Alpha Ranking...")

        # 1. Fetch The Universe (Stock Data)
        # We fetch full OHLCV for validation, plus features for training
        query = f"""
            SELECT 
                fd.*, 
                mdd.{MarketCol.OPEN},
                mdd.{MarketCol.HIGH},
                mdd.{MarketCol.LOW},
                mdd.{MarketCol.CLOSE},
                mdd.{MarketCol.VOLUME},
                a.{MarketCol.SYMBOL}
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{self.config.start_date}' AND fd.time <= '{self.config.end_date}'
            ORDER BY fd.time ASC
        """

        try:
            df = pl.read_database_uri(query, self.db_url)

            # --- UNIVERSE FILTERS ---
            # A. Price > $5
            df = df.filter(pl.col(MarketCol.CLOSE) >= 5.0)

            # B. Liquidity > $2M/day
            if "volume_adv_20" in df.columns:
                df = df.filter(
                    (pl.col(MarketCol.CLOSE) * pl.col("volume_adv_20")) > 2_000_000
                )

        except Exception as e:
            self.logger.error(f"DB Error: {e}")
            return pl.DataFrame()

        if df.is_empty():
            return df

        # 2. Apply Regime Logic (The Critical Step)
        if self.regime_labeler:
            self.logger.info("Applying regime labels via injected labeler...")

            # We pass the STOCK dataframe to the labeler.
            # The labeler (updated below) will be smart enough to fetch
            # the missing MARKET CONTEXT (Breadth) it needs from the DB.
            df = self.regime_labeler.label(df)

            # Filter rows based on configured regime(s)
            if self.config.filter_regime is not None and RegimeCol.TARGET in df.columns:
                target_regimes = self.config.filter_regime
                if isinstance(target_regimes, int):
                    target_regimes = [target_regimes]

                original_count = len(df)
                df = df.filter(pl.col(RegimeCol.TARGET).is_in(target_regimes))
                self.logger.info(
                    f"Regime Filter ({target_regimes}): {original_count} -> {len(df)} rows"
                )

                if df.is_empty():
                    self.logger.warning("Dataset is empty after regime filtering!")
                    return df

        # 3. Calculate Target (Forward Return)
        horizon = self.config.target_horizon_days
        target_col = self.config.target_column

        df = df.with_columns(
            ((pl.col(MarketCol.CLOSE).shift(-horizon) / pl.col(MarketCol.CLOSE)) - 1)
            .over(MarketCol.ASSET_ID)
            .alias(target_col)
        )

        return df.drop_nulls()
