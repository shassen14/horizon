import argparse
import polars as pl
from urllib.parse import quote_plus

from packages.data_pipelines.labeling.logic import RegimeLabeler
from packages.contracts.blueprints import LabelingConfig
from packages.contracts.vocabulary.columns import MarketCol
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager


class LabelingEngine:
    """
    Orchestrator for the offline regime labeling process.

    Responsibilities:
    1. Connect to the database to fetch raw price data.
    2. Instantiate and run the RegimeLabeler logic.
    3. Save the resulting labels to a Parquet artifact file.
    """

    def __init__(self, logger=None):
        self.logger = logger or LogManager("labeling-engine", debug=True).get_logger(
            "main"
        )

        # Define the artifact directory relative to this file's location
        self.artifacts_dir = settings.system.ARTIFACTS_ROOT / "labeling"

        # Ensure the directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run(self, horizon: int, config: LabelingConfig = LabelingConfig()):
        """
        Executes the full labeling pipeline for a given time horizon.

        Args:
            horizon (int): The forward-looking window in days (e.g., 21 for tactical, 63 for structural).
            config (LabelingConfig): Pydantic model with GMM and smoothing parameters.
        """
        self.logger.info(f"--- Starting Label Generation (Horizon: {horizon}d) ---")

        # 1. Load Data (I/O)
        try:
            df = self._load_spy_data("2000-01-01")
            if df.is_empty():
                self.logger.error("No data found for SPY. Aborting.")
                return
        except Exception as e:
            self.logger.error(f"Failed to load data from database: {e}", exc_info=True)
            return

        # 2. Instantiate and Execute Pure Logic
        self.logger.info("Calculating forward features and fitting GMM...")
        labeler = RegimeLabeler(config, horizon)

        try:
            labeled_df = labeler.fit_predict(df, price_col=MarketCol.CLOSE)
        except ValueError as e:
            self.logger.error(f"Labeling logic failed: {e}")
            return
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during GMM fitting: {e}", exc_info=True
            )
            return

        # 3. Save Artifacts (I/O)
        filename = f"regime_labels_{horizon}d.parquet"
        output_path = self.artifacts_dir / filename

        try:
            labeled_df.write_parquet(output_path)
            self.logger.success(f"✅ Saved {len(labeled_df)} labels to {output_path}")
        except Exception as e:
            self.logger.error(
                f"Failed to write artifact file to {output_path}: {e}", exc_info=True
            )

    def _load_spy_data(self, start_date: str) -> pl.DataFrame:
        """Connects to the database and fetches the close price history for SPY."""
        self.logger.info(
            f"Fetching SPY price history from database (start: {start_date})..."
        )

        safe_password = quote_plus(settings.db.password)
        db_url = f"postgresql://{settings.db.user}:{safe_password}@{settings.db.host}:{settings.db.port}/{settings.db.name}"

        query = f"""
            SELECT 
                mdd.{MarketCol.TIME}, 
                mdd.{MarketCol.CLOSE}
            FROM market_data_daily mdd
            JOIN asset_metadata a ON mdd.asset_id = a.id
            WHERE a.symbol = 'SPY' AND mdd.time >= '{start_date}'
            ORDER BY mdd.time ASC
        """
        return pl.read_database_uri(query, db_url)


if __name__ == "__main__":
    # This allows the engine to be run directly from the command line
    parser = argparse.ArgumentParser(
        description="Generate Market Regime Labels Offline"
    )
    parser.add_argument(
        "horizons",
        type=int,
        nargs="+",  # Allows one or more horizons (e.g., `... engine.py 21 63`)
        help="The forward-looking horizon(s) in days for labeling.",
    )

    args = parser.parse_args()

    engine = LabelingEngine()

    for h in args.horizons:
        engine.run(horizon=h)
