# packages/ml_core/analysis/regime_comparison.py

import polars as pl
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager


def compare_regime_methods():
    lm = LogManager("analysis", debug=True)
    logger = lm.get_logger("main")

    # 1. Direct DB Connection
    safe_password = quote_plus(settings.db.password)
    db_url = (
        f"postgresql://{settings.db.user}:{safe_password}@"
        f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
    )

    # 2. Query SPY Directly (Raw Price Data)
    logger.info("Fetching SPY history from DB...")
    query = """
        SELECT 
            mdd.time,
            mdd.close as close_price
        FROM market_data_daily mdd
        JOIN asset_metadata a ON mdd.asset_id = a.id
        WHERE a.symbol = 'SPY'
        AND mdd.time >= '1999-01-01'
        ORDER BY mdd.time ASC
    """

    try:
        df = pl.read_database_uri(query, db_url)
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return

    if df.is_empty():
        logger.error("No SPY data found.")
        return

    logger.info(f"Loaded {len(df)} rows. Calculating features...")

    # 3. Calculate Features for Clustering
    # Target: Forward 21-day Return and Forward 21-day Volatility
    horizon = 21

    df = df.with_columns(
        [
            # Forward Return: (Price in 21 days / Price Today) - 1
            # shift(-21) gets the future value
            ((pl.col("close_price").shift(-horizon) / pl.col("close_price")) - 1).alias(
                "fwd_ret"
            ),
            # Forward Volatility: StdDev of returns over next 21 days
            pl.col("close_price")
            .pct_change()
            .rolling_std(window_size=horizon)
            .shift(-horizon)
            .alias("fwd_vol"),
        ]
    ).drop_nulls()

    # 4. Prepare X (Outcomes)
    # Convert to Pandas for Scikit-Learn
    X = df.select(["fwd_ret", "fwd_vol"]).to_pandas()

    # Clean infinite values if any
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # Normalize (Crucial for GMM/HMM)
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    # --- A. Gaussian Mixture Model (Memoryless) ---
    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)

    # --- B. Hidden Markov Model (Stateful) ---
    print("Fitting HMM...")
    hmm = GaussianHMM(
        n_components=3, covariance_type="full", n_iter=100, random_state=42
    )
    hmm.fit(X_scaled)
    hmm_labels = hmm.predict(X_scaled)

    # --- 5. Align Labels ---
    # Sort labels so 0=Bull (Low Vol), 2=Bear (High Vol)
    def align_labels(labels, vol_data):
        df_temp = pd.DataFrame({"label": labels, "vol": vol_data})
        # Calculate mean vol per cluster
        avg_vol = df_temp.groupby("label")["vol"].mean().sort_values()
        # Map old_label -> new_label (0, 1, 2 based on volatility)
        mapping = {old: new for new, old in enumerate(avg_vol.index)}
        return np.array([mapping[l] for l in labels])

    gmm_sorted = align_labels(gmm_labels, X["fwd_vol"])
    hmm_sorted = align_labels(hmm_labels, X["fwd_vol"])

    # --- 6. Plotting ---
    dates = df["time"].to_pandas().iloc[X.index]
    prices = df["close_price"].to_pandas().iloc[X.index]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Price Chart
    axes[0].plot(dates, prices, color="black", alpha=0.6, linewidth=1)
    axes[0].set_title("SPY Price History")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # GMM Chart
    axes[1].scatter(dates, prices, c=gmm_sorted, cmap="RdYlGn_r", s=2)
    axes[1].set_title("GMM Regimes (Memoryless) - Green=Low Vol, Red=High Vol")
    axes[1].grid(True, alpha=0.3)

    # HMM Chart
    axes[2].scatter(dates, prices, c=hmm_sorted, cmap="RdYlGn_r", s=2)
    axes[2].set_title("HMM Regimes (Stateful) - Green=Low Vol, Red=High Vol")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "regime_comparison.png"
    plt.savefig(output_file)
    logger.success(f"Chart saved to {output_file}")


if __name__ == "__main__":
    compare_regime_methods()
