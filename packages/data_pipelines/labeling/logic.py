import polars as pl
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from packages.contracts.blueprints import LabelingConfig
from packages.contracts.vocabulary.columns import MarketCol, RegimeCol


class RegimeLabeler:
    """
    Pure Logic Component.
    Responsibility: Take a DataFrame with Price history, return a DataFrame with Regime Labels.
    """

    def __init__(self, config: LabelingConfig, horizon: int):
        self.cfg = config
        self.horizon = horizon

    def fit_predict(self, df: pl.DataFrame, price_col: str = "close") -> pl.DataFrame:
        """
        1. Calculates Forward Returns/Vol
        2. Fits GMM
        3. Sorts Clusters (0=LowVol, N=HighVol)
        4. Smooths Signals
        """
        # 1. Feature Engineering (Outcomes)
        # We need to compute what happens *next* to label *now*.
        lab_df = (
            df.select([pl.col(MarketCol.TIME), pl.col(price_col).alias("temp_close")])
            .with_columns(
                [
                    (
                        (
                            pl.col("temp_close").shift(-self.horizon)
                            / pl.col("temp_close")
                        )
                        - 1
                    ).alias(RegimeCol.FWD_RET),
                    pl.col("temp_close")
                    .pct_change()
                    .rolling_std(self.horizon)
                    .shift(-self.horizon)
                    .alias(RegimeCol.FWD_VOL),
                ]
            )
            .drop_nulls()
        )

        if lab_df.height < (self.cfg.n_clusters * 10):
            raise ValueError("Not enough data to form stable clusters.")

        # 2. Prepare X for GMM
        # Convert to Pandas/Numpy for Scikit-Learn
        X = lab_df.select(self.cfg.forward_cols).to_pandas()

        # Normalize (Critical for GMM convergence)
        X_norm = (X - X.mean()) / X.std()
        X_norm = X_norm.fillna(0)

        # 3. Fit GMM
        gmm = GaussianMixture(
            n_components=self.cfg.n_clusters,
            random_state=42,  # Fixed seed for reproducibility
            n_init=5,  # Run 5 times, keep best
        )
        raw_labels = gmm.fit_predict(X_norm)

        # 4. Semantic Sorting (The "Magic" Fix)
        # We enforce that Label 0 is always the Lowest Volatility state.
        # This ensures consistency between runs.
        cluster_vols = []
        for c in range(self.cfg.n_clusters):
            mask = raw_labels == c
            if mask.any():
                # Use the correct Enum column name to access the data
                vol = lab_df[RegimeCol.FWD_VOL].to_numpy()[mask].mean()
                cluster_vols.append(vol)
            else:
                cluster_vols.append(999.0)

        # Create mapping: argsort gives indices that would sort the array
        # e.g. Vols = [0.05, 0.01, 0.10] -> argsort -> [1, 0, 2]
        sorted_indices = np.argsort(cluster_vols)

        # Map: Old Label -> New Label
        # Old 1 (0.01) becomes New 0
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        sorted_labels = np.array([mapping[l] for l in raw_labels])

        # 5. Persistence (Smoothing)
        # Prevent "Flickering" by taking the mode of the last N days
        final_labels = self._apply_smoothing(sorted_labels)

        # 6. Attach back to Polars
        result = lab_df.with_columns(
            pl.Series(RegimeCol.TARGET, final_labels).cast(pl.Int32)
        )

        # Return only Time and Label (The contract)
        return result.select([MarketCol.TIME, RegimeCol.TARGET])

    def _apply_smoothing(self, labels: np.ndarray) -> np.ndarray:
        """Rolling Mode filter to reduce noise."""
        if self.cfg.smoothing_window <= 1:
            return labels

        s = pd.Series(labels)
        # centered=True means we look ahead and behind, but for labeling historical data that's fine.
        # For training targets, we want the "True" state, so centered smoothing is acceptable.
        smoothed = s.rolling(window=self.cfg.smoothing_window, center=True).apply(
            lambda x: x.mode()[0] if not x.mode().empty else x[0], raw=False
        )
        return smoothed.ffill().bfill().astype(int).values
