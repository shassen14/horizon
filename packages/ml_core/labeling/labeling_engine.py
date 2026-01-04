# packages/ml_core/labeling/labeling_engine.py

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from urllib.parse import quote_plus

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager


class LabelingEngine:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            lm = LogManager("labeling-engine", debug=True)
            self.logger = lm.get_logger("main")

        self.artifacts_dir = (
            Path(__file__).resolve().parents[1] / "labeling" / "artifacts"
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def generate_labels(self, horizon: int = 63, n_clusters: int = 3):
        """
        Generates regime labels using Unsupervised Learning (GMM) on Forward Outcomes.

        Args:
            horizon (int): The forward-looking window (e.g. 63 for Structural, 21 for Tactical).
            n_clusters (int): Number of regimes (3 is standard: Bull, Chop, Bear).
        """
        self.logger.info(
            f"--- Generating Labels for Horizon: {horizon}d (Clusters: {n_clusters}) ---"
        )

        # 1. Load Raw SPY Data (Direct DB Query for speed)
        df = self._load_spy_data("2000-01-01")  # Long history for better clustering

        if df.is_empty():
            self.logger.error("No SPY data found.")
            return

        # 2. Feature Engineering (Forward Looking Outcomes)
        # We define the regime by what HAPPENED, not what predicted it.

        df = df.with_columns(
            [
                # A. Forward Return
                ((pl.col("close").shift(-horizon) / pl.col("close")) - 1).alias(
                    "fwd_ret"
                ),
                # B. Forward Volatility (Standard Deviation of returns over the horizon)
                pl.col("close")
                .pct_change()
                .rolling_std(window_size=horizon)
                .shift(-horizon)
                .alias("fwd_vol"),
                # C. Forward Max Drawdown (Optional but powerful - max loss in the window)
                # (Simplified here to just Ret/Vol for robustness, but could add DD)
            ]
        ).drop_nulls()

        # 3. Prepare X matrix for Clustering
        X = df.select(["fwd_ret", "fwd_vol"]).to_pandas()

        # Remove outliers/inf
        X = X.replace([np.inf, -np.inf], np.nan).dropna()

        # Normalize (Crucial for GMM)
        X_mean, X_std = X.mean(), X.std()
        X_scaled = (X - X_mean) / X_std

        # 4. Fit GMM
        self.logger.info("Fitting Gaussian Mixture Model...")
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type="full", random_state=42
        )
        raw_labels = gmm.fit_predict(X_scaled)

        # 5. Semantic Sorting
        # GMM labels are random (0 might be Bear). We force order by Volatility.
        # 0 = Low Vol (Bull), 2 = High Vol (Bear)
        df_temp = pd.DataFrame({"label": raw_labels, "vol": X["fwd_vol"]})
        avg_vol = df_temp.groupby("label")["vol"].mean().sort_values()

        mapping = {old: new for new, old in enumerate(avg_vol.index)}
        sorted_labels = np.array([mapping[l] for l in raw_labels])

        # 6. Apply Persistence (Smoothing)
        # Prevent "Flicker" where regime jumps 0->1->0 in 2 days.
        smoothed_labels = self._enforce_persistence(sorted_labels, window=5)

        # 7. Save Artifacts
        # Map back to dates
        output_df = df.select("time").to_pandas().iloc[X.index]  # Align indices
        output_df["regime_label"] = smoothed_labels

        # Save Parquet: regime_labels_63d.parquet
        filename = f"regime_labels_{horizon}d.parquet"
        output_path = self.artifacts_dir / filename
        output_df.to_parquet(output_path, index=False)

        self.logger.info(f"Regime Stats (Sorted 0=LowVol -> {n_clusters-1}=HighVol):")
        df_temp["sorted_label"] = sorted_labels
        self.logger.info(df_temp.groupby("sorted_label")["vol"].describe())

        self.logger.success(f"Labels saved to {output_path}")

    def _enforce_persistence(self, labels: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Block D: Persistence.
        Uses a rolling mode (majority vote) to smooth out noise.
        """
        s = pd.Series(labels)
        # Calculate rolling mode. If window=5, needs 3 days to confirm a switch.
        smoothed = s.rolling(window=window, center=True).apply(
            lambda x: x.mode()[0] if not x.mode().empty else x[0], raw=False
        )
        return smoothed.fillna(method="ffill").fillna(method="bfill").astype(int).values

    def _load_spy_data(self, start_date):
        safe_password = quote_plus(settings.db.password)
        db_url = f"postgresql://{settings.db.user}:{safe_password}@{settings.db.host}:{settings.db.port}/{settings.db.name}"

        query = f"""
            SELECT mdd.time, mdd.close
            FROM market_data_daily mdd
            JOIN asset_metadata a ON mdd.asset_id = a.id
            WHERE a.symbol = 'SPY' AND mdd.time >= '{start_date}'
            ORDER BY mdd.time ASC
        """
        return pl.read_database_uri(query, db_url)


if __name__ == "__main__":
    engine = LabelingEngine()

    # Run for Structural (63d)
    engine.generate_labels(horizon=63, n_clusters=3)

    # Run for Tactical (21d)
    engine.generate_labels(horizon=21, n_clusters=3)
