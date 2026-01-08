import polars as pl
from typing import Optional

# --- Abstraction Import ---
from packages.data_pipelines.labeling.base import AbstractLabeler

# --- Implementation Import ---
from packages.ml_ops.registry_client import RegistryClient
from packages.ml_ops.modeling.pipeline import HorizonPipeline

# --- Contract Imports ---
from packages.contracts.vocabulary.columns import MarketCol, RegimeCol


class RegimeModelLabeler(AbstractLabeler):
    """
    A concrete implementation of AbstractLabeler that uses a trained Regime Model
    from the MLflow Registry to label a DataFrame.
    """

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
        self.logger = logger

        # Load the pipeline during initialization
        self.pipeline: Optional[HorizonPipeline] = self._load_model()

    def _load_model(self) -> Optional[HorizonPipeline]:
        """Loads the specified regime model pipeline from the registry."""
        if self.logger:
            self.logger.info(
                f"Loading dependency: Regime Model '{self.model_name}@{self.alias}'..."
            )

        pipeline = self.registry_client.load_pipeline(self.model_name, self.alias)

        if not pipeline and self.logger:
            self.logger.warning(
                f"Regime model '{self.model_name}@{self.alias}' not found in registry."
            )

        return pipeline

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies regime labels to the input DataFrame.
        If the model fails to load, it applies a default 'safe' label.
        """
        if not self.pipeline:
            # Fallback: If the regime model is missing, assume a default "safe" regime (e.g., Bull=1)
            if self.logger:
                self.logger.warning(
                    "Applying default regime label (1) due to missing model."
                )
            return df.with_columns(pl.lit(1, dtype=pl.Int32).alias(RegimeCol.TARGET))

        try:
            # 1. Reconstruct the exact feature vector the regime model expects.
            # This logic must be kept in sync with how the regime model was trained.
            # For now, we assume it's based on market-wide aggregates.
            self.logger.info("Reconstructing features for regime model prediction...")

            # We need to ensure the necessary base features are present
            required_cols = [
                "sma_50",
                "atr_14_pct",
                "rsi_14",
                "zscore_20",
                MarketCol.CLOSE,
            ]
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(
                    f"Input DataFrame is missing required columns for regime labeling: {missing}"
                )

            market_stats = (
                df.group_by(MarketCol.TIME)
                .agg(
                    [
                        ((pl.col(MarketCol.CLOSE) > pl.col("sma_50")).mean()).alias(
                            "pct_above_sma50"
                        ),
                        pl.col("atr_14_pct").mean().alias("avg_market_vol"),
                    ]
                )
                .sort(MarketCol.TIME)
            )

            # Isolate SPY features
            spy_df = (
                df.filter(pl.col(MarketCol.SYMBOL) == "SPY")
                .select([MarketCol.TIME, "rsi_14", "zscore_20"])
                .rename({"rsi_14": "spy_rsi_14", "zscore_20": "spy_zscore_20"})
            )

            regime_input = market_stats.join(
                spy_df, on=MarketCol.TIME, how="left"
            ).drop_nulls()

            # 2. Predict
            if self.logger:
                self.logger.info(f"Predicting regimes for {len(regime_input)} days...")
            regime_preds = self.pipeline.predict(regime_input)

            # 3. Map predictions back to a time-indexed series
            regime_labels = regime_input.select(MarketCol.TIME).with_columns(
                pl.Series(name=RegimeCol.TARGET, values=regime_preds, dtype=pl.Int32)
            )

            # 4. Broadcast the daily regime label to all stocks on that day
            return df.join(regime_labels, on=MarketCol.TIME, how="left")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to apply regime labels: {e}", exc_info=True)
            # Fallback on error
            return df.with_columns(pl.lit(1, dtype=pl.Int32).alias(RegimeCol.TARGET))
