import polars as pl
from .base import AbstractDatasetBuilder
from packages.contracts.vocabulary.columns import MarketCol
from packages.quant_lib.config import settings


class RegimeDatasetBuilder(AbstractDatasetBuilder):
    """
    Builds the regime dataset by joining SPY price action with
    the pre-calculated Market Context table.
    """

    def _load_data_internal(self) -> pl.DataFrame:
        self.logger.info(f"Building Regime Dataset...")

        # 1. Load SPY OHLCV (The Anchor and Target basis)
        spy_df = self._load_spy_ohlcv()
        if spy_df.is_empty():
            raise ValueError("SPY data missing, cannot build regime dataset.")

        # --- FIX: Normalize SPY Time ---
        spy_df = spy_df.with_columns(pl.col("time").dt.truncate("1d"))

        # 2. Load Market Context (Features)
        context_df = self._load_market_context()
        # --- FIX: Normalize Context Time ---
        context_df = context_df.with_columns(pl.col("time").dt.truncate("1d"))

        # 3. Join
        df = spy_df.join(context_df, on=MarketCol.TIME, how="inner").sort(
            MarketCol.TIME
        )

        # 4. Attach Labels (Ground Truth)
        df = self._attach_labels(df)

        return df.drop_nulls()

    def _load_spy_ohlcv(self) -> pl.DataFrame:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM market_data_daily JOIN asset_metadata a ON asset_id = a.id
            WHERE a.symbol = 'SPY' AND time >= '{self.config.start_date}'
        """
        return pl.read_database_uri(query, self.db_url)

    def _load_market_context(self) -> pl.DataFrame:
        query = f"""
            SELECT * FROM market_context_daily WHERE time >= '{self.config.start_date}'
        """
        return pl.read_database_uri(query, self.db_url)

    def _attach_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        from pathlib import Path

        horizon = self.config.target_horizon_days
        filename = f"regime_labels_{horizon}d.parquet"
        label_path = settings.system.ARTIFACTS_ROOT / "labeling" / filename

        if not label_path.exists():
            raise FileNotFoundError(f"Labels file missing: {filename}.")

        labels = pl.read_parquet(label_path).with_columns(
            pl.col("time").dt.truncate("1d")
        )
        return df.join(labels, on=MarketCol.TIME, how="inner")
