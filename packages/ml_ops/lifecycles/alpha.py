import polars as pl
import pandas as pd
from packages.ml_ops.protocols import ModelLifecycle, ValidationLogic
from packages.ml_ops.validation.permutators.ohlc import OHLCPermutator
from packages.contracts.vocabulary.columns import MarketCol


class AlphaValidationLogic(ValidationLogic):
    def permute_data(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        # Multi-Asset Logic: Permute each symbol independently
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

        grouped = df.groupby(MarketCol.SYMBOL)
        chunks = []
        for sym, group in grouped:
            if len(group) > 50:
                # Deterministic seed per asset
                asset_seed = (seed + hash(sym)) % (2**32)
                chunks.append(permutator.permute(group, seed=asset_seed))

        return pd.concat(chunks) if chunks else pd.DataFrame()

    def relabel_data(self, df: pl.DataFrame, config) -> pl.DataFrame:
        # Simple Forward Return calculation
        horizon = config.target_horizon_days
        target_col = config.target_column

        return df.with_columns(
            ((pl.col(MarketCol.CLOSE).shift(-horizon) / pl.col(MarketCol.CLOSE)) - 1)
            .over(MarketCol.ASSET_ID)  # Essential for multi-asset
            .alias(target_col)
        ).drop_nulls(subset=[target_col])


class AlphaLifecycle(ModelLifecycle):
    def prepare_training_data(self, raw_df: pl.DataFrame, config) -> pl.DataFrame:
        # Alpha Builder calculates target, but we can enforce it here
        return raw_df

    def get_validation_logic(self) -> ValidationLogic:
        return AlphaValidationLogic()
