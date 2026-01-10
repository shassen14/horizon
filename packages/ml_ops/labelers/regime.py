import polars as pl
from typing import Optional

# --- Abstraction Import ---
from packages.data_pipelines.labeling.base import AbstractLabeler

# --- Implementation Import ---
from packages.ml_ops.registry_client import RegistryClient
from packages.ml_ops.modeling.pipeline import HorizonPipeline

# --- Contract Imports ---
from packages.contracts.vocabulary.columns import MarketCol, RegimeCol
from packages.contracts.blueprints import RegimeDataConfig

# --- Builder Import ---
from packages.data_pipelines.builders.regime import RegimeDatasetBuilder
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager


class RegimeModelLabeler(AbstractLabeler):
    def __init__(
        self,
        registry_client: RegistryClient,
        model_name: str,
        alias: str = "production",
        logger=None,
    ):
        self.registry_client = registry_client
        self.model_name = model_name
        self.alias = alias
        self.logger = logger or LogManager("regime-labeler").get_logger("main")
        self.pipeline: Optional[HorizonPipeline] = self._load_model()

    def _load_model(self) -> Optional[HorizonPipeline]:
        pipeline = self.registry_client.load_pipeline(self.model_name, self.alias)
        if not pipeline and self.logger:
            self.logger.warning(f"Regime model '{self.model_name}' not found.")
        return pipeline

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.pipeline:
            return df.with_columns(pl.lit(1, dtype=pl.Int32).alias(RegimeCol.TARGET))

        try:
            # 1. Reuse the RegimeDatasetBuilder Logic
            # We construct a configuration that matches the Alpha data range
            min_date = df[MarketCol.TIME].min().strftime("%Y-%m-%d")

            # Create a temporary config for the builder
            # We hardcode '21' here assuming the model is tactical, but ideally
            # we should read this from the pipeline metadata if stored.
            # For now, 21d is a safe default for fetching the base data.
            regime_config = RegimeDataConfig(
                dataset_builder="RegimeDatasetBuilder",
                start_date=min_date,
                target_horizon_days=21,
            )

            self.logger.info("Fetching market context using RegimeDatasetBuilder...")

            # Instantiate Builder
            # We pass a logger to silence it if we want, or reuse self.logger
            builder = RegimeDatasetBuilder(settings, self.logger, regime_config)

            # Load Data (This uses the smart, correct SQL logic we already wrote)
            # Note: We use _load_data_internal to skip the file cache check,
            # because the date range might be a subset of a cached file.
            context_df = builder.get_data()  # Hits Parquet Cache!

            # 2. Align Features
            # The builder returns exactly what we need (SPY + Breadth + Technicals)
            # We just need to ensure column names match what the pipeline expects.

            # Drop the label column if it exists (we are predicting it afresh)
            if RegimeCol.TARGET in context_df.columns:
                context_df = context_df.drop(RegimeCol.TARGET)

            # 3. Predict
            regime_preds = self.pipeline.predict(context_df)

            # 4. Map back
            regime_labels = context_df.select(MarketCol.TIME).with_columns(
                pl.Series(name=RegimeCol.TARGET, values=regime_preds, dtype=pl.Int32)
            )

            # 5. Broadcast to Alpha Dataset
            return df.join(regime_labels, on=MarketCol.TIME, how="left")

        except Exception as e:
            self.logger.error(f"Regime labeling failed: {e}", exc_info=True)
            return df.with_columns(pl.lit(1, dtype=pl.Int32).alias(RegimeCol.TARGET))
