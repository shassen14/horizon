import polars as pl
import pandas as pd
from packages.ml_ops.protocols import ModelLifecycle, ValidationLogic
from packages.data_pipelines.labeling.logic import RegimeLabeler
from packages.ml_ops.validation.permutators.ohlc import OHLCPermutator
from packages.contracts.vocabulary.columns import MarketCol


class RegimeValidationLogic(ValidationLogic):
    def permute_data(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        # Standard Synchronous Permutation for Single Asset
        # Identify "Extra" columns (Breadth, Macro) to permute via value shuffle
        ohlc_cols = {
            MarketCol.OPEN,
            MarketCol.HIGH,
            MarketCol.LOW,
            MarketCol.CLOSE,
            MarketCol.VOLUME,
            MarketCol.SYMBOL,
            MarketCol.TIME,
            MarketCol.ASSET_ID,
        }
        extra_cols = [c for c in df.columns if c not in ohlc_cols]

        permutator = OHLCPermutator(diff_cols=extra_cols)
        return permutator.permute(df, seed=seed)

    def relabel_data(self, df: pl.DataFrame, config) -> pl.DataFrame:
        # Re-run GMM on the new price path
        price_col = "SPY" if "SPY" in df.columns else MarketCol.CLOSE
        labeler = RegimeLabeler(config.labeling, config.target_horizon_days)

        # Calculate labels
        labels = labeler.fit_predict(df, price_col=price_col)
        # Join back
        return df.join(labels, on=MarketCol.TIME, how="inner")


class RegimeLifecycle(ModelLifecycle):
    def prepare_training_data(self, raw_df: pl.DataFrame, config) -> pl.DataFrame:
        # For Regime, the Builder already attached labels from the artifacts.
        # So we just ensure types and return.
        # (If we wanted to run labeling live during training, we'd do it here).
        return raw_df

    def get_validation_logic(self) -> ValidationLogic:
        return RegimeValidationLogic()
