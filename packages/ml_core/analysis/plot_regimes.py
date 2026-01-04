import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from urllib.parse import quote_plus

# We need to import settings to get the DB URL
from packages.quant_lib.config import settings


def plot_regime_labels():
    """
    Loads SPY price data and the generated regime labels, then plots them
    on a chart to visually validate the clustering.
    """
    print("--- Generating Regime Validation Chart ---")

    # --- 1. Define Paths ---
    artifacts_dir = Path(__file__).resolve().parents[1] / "labeling" / "artifacts"

    # Paths to your ground truth files
    path_63d = artifacts_dir / "regime_labels_63d.parquet"
    path_21d = artifacts_dir / "regime_labels_21d.parquet"
    output_path = "regime_label_validation.png"

    # --- 2. Load SPY Price Data ---
    safe_password = quote_plus(settings.db.password)
    db_url = (
        f"postgresql://{settings.db.user}:{safe_password}@"
        f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
    )
    query = """
        SELECT mdd.time, mdd.close
        FROM market_data_daily mdd
        JOIN asset_metadata a ON mdd.asset_id = a.id
        WHERE a.symbol = 'SPY'
        ORDER BY mdd.time ASC
    """

    try:
        spy_df = pl.read_database_uri(query, db_url)
        labels_63d = pl.read_parquet(path_63d)
        labels_21d = pl.read_parquet(path_21d)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 3. Load Label Data ---
    spy_df = spy_df.with_columns(pl.col("time").dt.date().alias("date"))
    labels_63d = labels_63d.with_columns(pl.col("time").dt.date().alias("date"))
    labels_21d = labels_21d.with_columns(pl.col("time").dt.date().alias("date"))

    # --- 4. Join Data ---
    # Merge price data with both sets of labels
    plot_df = spy_df.join(labels_63d, on="time", how="left")
    plot_df = plot_df.join(labels_21d, on="time", how="left", suffix="_21d")
    plot_df = plot_df.rename(
        {"regime_label": "regime_63d", "regime_label_21d": "regime_21d"}
    )

    #  Perform the join on the new 'date' column
    # We use 'inner' join to ensure we only plot where we have both price and label
    plot_df = spy_df.join(labels_63d, on="date", how="inner")
    plot_df = plot_df.join(
        labels_21d.select(["date", "regime_label"]),
        on="date",
        how="inner",
        suffix="_21d",
    )

    plot_df = plot_df.rename(
        {"regime_label": "regime_63d", "regime_label_21d": "regime_21d"}
    )

    # Convert to Pandas for easier plotting with matplotlib
    plot_pdf = plot_df.to_pandas()
    plot_pdf["time"] = pd.to_datetime(plot_pdf["time"])

    if plot_pdf["regime_63d"].isnull().any() or plot_pdf["regime_21d"].isnull().any():
        print(
            "ERROR: Null values still present in regime columns after join. Aborting plot."
        )
        return

    # --- 5. Create the Plot ---
    # We will create 3 subplots: Price, Structural Regimes, Tactical Regimes
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

    # Plot 1: SPY Price (Log Scale to see crashes better)
    axes[0].plot(
        plot_pdf["time"], plot_pdf["close"], color="black", alpha=0.7, linewidth=1.5
    )
    axes[0].set_title("SPY Price History (Log Scale)", fontsize=16)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # Helper for coloring
    # We assume 0=Green (Bull), 1=Yellow (Chop), 2=Red (Bear/Crisis)
    cmap = {0: "green", 1: "gold", 2: "red"}

    # Plot 2: Structural Regimes (63d)
    # We use a scatter plot, coloring each point by its regime label
    axes[1].scatter(
        plot_pdf["time"],
        plot_pdf["close"],
        c=plot_pdf["regime_63d"].map(cmap),
        s=5,  # Small dot size
        alpha=0.8,
    )
    axes[1].set_title("Structural Regimes (63d)", fontsize=16)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    # Plot 3: Tactical Regimes (21d)
    axes[2].scatter(
        plot_pdf["time"],
        plot_pdf["close"],
        c=plot_pdf["regime_21d"].map(cmap),
        s=5,
        alpha=0.8,
    )
    axes[2].set_title("Tactical Regimes (21d)", fontsize=16)
    axes[2].set_yscale("log")
    axes[2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)

    print(f"✅ Chart saved successfully to: {output_path}")


if __name__ == "__main__":
    # Ensure PYTHONPATH is set correctly when running this
    plot_regime_labels()
