import mlflow
import polars as pl
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus
from sqlalchemy import select, func, text, update
from sqlalchemy.dialects.postgresql import insert

from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from packages.ml_core.modeling.pipeline import HorizonPipeline
from packages.database.models import (
    Asset,
    MarketDataDaily,
    FeaturesDaily,
    Model,
    Prediction,
)
from packages.database.session import get_db_session


class InferenceEngine:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            # Create a default logger if one isn't provided
            lm = LogManager("inference-engine", settings.system.debug)
            self.logger = lm.get_logger("main")

        # 1. Initialize Connection to MLflow Registry
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

        # In-memory cache for loaded models to avoid disk I/O on every run
        self._model_cache: Dict[str, HorizonPipeline] = {}

        # Synchronous DB URL for Polars' high-speed connector
        safe_password = quote_plus(settings.db.password)
        self.db_url = (
            f"postgresql://{settings.db.user}:{safe_password}@"
            f"{settings.db.host}:{settings.db.port}/{settings.db.name}"
        )

    def _load_pipeline(
        self, model_name: str, alias: str = "production"
    ) -> Optional[HorizonPipeline]:
        """Loads a model pipeline from the MLflow Registry using an Alias, with RAM caching."""
        cache_key = f"{model_name}_@{alias}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_uri = f"models:/{model_name}@{alias}"

        try:
            self.logger.info(f"Loading '{model_uri}' from Registry...")
            wrapper = mlflow.pyfunc.load_model(model_uri)
            pipeline = wrapper._model_impl.python_model.pipeline
            self._model_cache[cache_key] = pipeline
            return pipeline
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}@{alias}: {e}")
            return None

    async def get_market_regime(self) -> Dict[str, Any]:
        """Loads the Regime Model and predicts the current market state."""
        pipeline = self._load_pipeline("regime_classifier_v1", alias="production")

        if not pipeline:
            self.logger.warning("Regime model not found. Defaulting to Bull (1).")
            return {"regime": 1, "probability": 0.5}

        regime_df = await self._get_regime_input_data()

        if regime_df.is_empty():
            self.logger.warning("No data for regime prediction. Defaulting to Bull.")
            return {"regime": 1, "probability": 0.5}

        try:
            latest_row = regime_df.tail(1)
            pred_class = pipeline.predict(latest_row)[0]

            if hasattr(pipeline.model, "predict_proba"):
                X, _ = pipeline.get_X_y(latest_row)
                probs = pipeline.model.predict_proba(X)[0]
                bull_prob = probs[1]
            else:
                bull_prob = 1.0 if pred_class == 1 else 0.0

            return {"regime": int(pred_class), "probability": float(bull_prob)}
        except Exception as e:
            self.logger.error(f"Regime inference failed: {e}", exc_info=True)
            return {"regime": 1, "probability": 0.5}

    async def run_alpha_ranking(self) -> Optional[pl.DataFrame]:
        """
        Main entry point for generating daily stock rankings.
        Returns a Polars DataFrame with prediction results.
        """
        regime_info = await self.get_market_regime()
        is_bull = regime_info["regime"] == 1

        model_name = "alpha_bull_v1" if is_bull else "alpha_bear_v1"
        self.logger.info(
            f"Market Regime: {'BULL' if is_bull else 'BEAR'}. Loading {model_name}."
        )

        pipeline = self._load_pipeline(model_name, alias="production")
        if not pipeline:
            self.logger.warning(f"Falling back to 'alpha_general_v1'.")
            pipeline = self._load_pipeline("alpha_general_v1", alias="production")
            if not pipeline:
                self.logger.error("No usable alpha models in Production.")
                return None

        df_history = await self._get_alpha_input_data(lookback_days=90)
        if df_history.is_empty():
            self.logger.error("No historical feature data for ranking.")
            return None

        # We filter out stocks that lack sufficient history to be stable.
        # Checking for 'sma_50' ensures the stock has traded for at least ~2.5 months.
        # If the strategy depends on long-term trends, require sma_200.
        # If it's medium-term, sma_50 is a good baseline.
        pre_count = df_history["symbol"].n_unique()

        # Drop Nulls (Maturity Check)
        df_clean = df_history.drop_nulls(
            subset=["sma_50", "rsi_14", "volume_adv_20", "high_252_pct"]
        )

        # Minimum Price (Avoid penny stocks)
        df_clean = df_clean.filter(pl.col("close") >= 5.0)

        # Falling Knife Check
        # high_252_pct is e.g., -0.90 for a 90% drop.
        # We want stocks that are NOT down more than 75% (-0.75).
        # We also handle cases where high_252_pct might be null or positive (rare)
        df_clean = df_clean.filter(pl.col("high_252_pct") > -0.75)

        #  Liquidity Check (Dollar Volume)
        # We want stocks trading > $2M per day to ensure they are real companies
        df_clean = df_clean.filter(
            (pl.col("close") * pl.col("volume_adv_20")) > 2_000_000
        )

        post_count = df_clean["symbol"].n_unique()
        if post_count < pre_count:
            self.logger.info(
                f"Quality Filter: Removed {pre_count - post_count} assets (Penny stocks / Falling Knives)."
            )

        if df_clean.is_empty():
            self.logger.warning("No assets remained after maturity filtering.")
            return None

        self.logger.info(f"Running inference on {post_count} assets...")

        try:
            scores = pipeline.predict(df_clean)

            df_scored = df_clean.with_columns(pl.Series(name="score", values=scores))

            latest_scores = df_scored.group_by("asset_id").tail(1)

            run_id = pipeline.run_id if pipeline.run_id else "unknown_run"

            final_ranks = (
                latest_scores.select(["time", "asset_id", "symbol", "close", "score"])
                .sort("score", descending=True)
                .with_columns(
                    [
                        pl.col("score")
                        .rank(descending=True, method="ordinal")
                        .alias("rank"),
                        pl.lit(regime_info["regime"]).alias("regime_used"),
                        # Store string metadata
                        pl.lit(run_id).alias("model_version"),
                        pl.lit(model_name).alias("model_name_used"),
                    ]
                )
            )

            self.logger.success(f"Generated rankings for {len(final_ranks)} assets.")
            return final_ranks
        except Exception as e:
            self.logger.error(f"Alpha inference failed: {e}", exc_info=True)
            return None

    async def save_predictions(self, df_ranks: pl.DataFrame):
        """
        Saves predictions and updates Model Registry using metadata EMBEDDED in the pipeline.
        No hardcoded strings required.
        """
        if df_ranks is None or df_ranks.is_empty():
            return

        # Get identifying info from the first row of predictions
        first_row = df_ranks.row(0, named=True)
        model_name = first_row["model_name_used"]
        model_version = first_row["model_version"]

        pipeline = None

        # Try to find it in the cache
        cache_key_prod = f"{model_name}_@production"

        if cache_key_prod in self._model_cache:
            pipeline = self._model_cache[cache_key_prod]
        else:
            # If not in cache (unlikely if we just ran inference), try to load it
            self.logger.info(
                f"Pipeline not in cache, reloading for metadata extraction..."
            )
            pipeline = self._load_pipeline(model_name, alias="production")

        # Extract Metadata
        if pipeline:
            description = pipeline.metadata.get("description", "Imported Model")

            # Construct the 'meta' JSON for the database
            # We filter out internal python-specific keys if any
            db_meta = pipeline.metadata.copy()

            # Add dynamic context if needed (e.g. generated time)
            db_meta["last_inference_run"] = datetime.now(timezone.utc).isoformat()
        else:
            self.logger.warning("Could not load pipeline metadata. Using defaults.")
            description = "Unknown Model"
            db_meta = {}

        self.logger.info(f"Saving predictions for {model_name} (v: {model_version})...")
        self.logger.info(f"Model Description: {description}")

        # 1. Upsert Model Definition
        async with get_db_session() as session:
            stmt = insert(Model).values(
                model_name=model_name,
                model_type="ALPHA",
                description=description,
                meta=db_meta,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["model_name"],
                set_={
                    "description": stmt.excluded.description,
                    "meta": stmt.excluded.meta,
                    "last_updated": func.now(),
                },
            )
            await session.execute(stmt)
            await session.commit()

        # 2. Bulk Upsert Predictions (Existing Logic)
        records = []
        for row in df_ranks.iter_rows(named=True):
            records.append(
                {
                    "time": row["time"],
                    "asset_id": row["asset_id"],
                    "model_name": model_name,
                    "model_version": model_version,
                    "output": {"score": row["score"], "rank": int(row["rank"])},
                }
            )

        async with get_db_session() as session:
            # Chunking 1000...
            for i in range(0, len(records), 1000):
                chunk = records[i : i + 1000]
                stmt = insert(Prediction).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["time", "asset_id", "model_name", "model_version"],
                    set_={"output": stmt.excluded.output},
                )
                await session.execute(stmt)
            await session.commit()

        self.logger.success("Predictions saved successfully.")

    async def _get_regime_input_data(self, lookback_days: int = 30) -> pl.DataFrame:
        """
        Reconstructs the feature set required by the regime model for the most recent data.
        We fetch 30 days to ensure we have a stable value for SMA50 and ATR.
        """
        self.logger.info("Fetching recent market-wide data for regime prediction...")
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        # 1. Query necessary base features for all active stocks
        # We need sma_50 and atr_14_pct for breadth, plus SPY's specific features
        query = f"""
            SELECT 
                fd.time, 
                fd.asset_id, 
                fd.sma_50, 
                fd.atr_14_pct,
                fd.rsi_14,      
                fd.zscore_20,
                mdd.close as close_price, 
                a.symbol
            FROM features_daily fd
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            JOIN asset_metadata a ON fd.asset_id = a.id
            WHERE fd.time >= '{start_date}'
            AND a.is_active = TRUE
        """

        try:
            full_df = pl.read_database_uri(query, self.db_url)
            if full_df.is_empty():
                self.logger.warning("No recent feature data found for regime input.")
                return pl.DataFrame()

            # 2. Calculate Market Breadth (Daily Aggregate)
            market_df = (
                full_df.group_by("time")
                .agg(
                    [
                        (
                            (pl.col("close_price") > pl.col("sma_50"))
                            .sum()
                            .cast(pl.Float64)
                            / pl.count()
                        ).alias("breadth_sma50_pct"),
                        pl.col("atr_14_pct").mean().alias("vol_market_avg_atr_pct"),
                    ]
                )
                .sort("time")
            )

            # 3. Get SPY's specific features
            spy_df = (
                full_df.filter(pl.col("symbol") == "SPY")
                .select(["time", "rsi_14", "zscore_20", "close_price"])
                .rename(
                    {
                        "rsi_14": "spy_rsi_14",
                        "zscore_20": "spy_zscore_20",
                        "close_price": "spy_close",
                    }
                )
            )

            if spy_df.is_empty():
                self.logger.warning(
                    "SPY data not found in recent features. Regime prediction may be inaccurate."
                )
                # We can proceed without SPY, the join will create nulls
                # which the model might handle if trained on such data.

            # 4. Join to create the final input DataFrame
            final_df = market_df.join(spy_df, on="time", how="left")

            # The model was trained with 'spy_daily_return'. We must generate it here.
            final_df = final_df.with_columns(
                pl.col("spy_close").pct_change().alias("spy_daily_return")
            )

            return final_df.fill_null(strategy="forward")

        except Exception as e:
            self.logger.error(f"Failed to load regime input data: {e}", exc_info=True)
            return pl.DataFrame()

    async def _get_alpha_input_data(self, lookback_days: int) -> pl.DataFrame:
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        query = f"""
            SELECT fd.*, a.symbol, mdd.close
            FROM features_daily fd
            JOIN asset_metadata a ON fd.asset_id = a.id
            JOIN market_data_daily mdd ON fd.time = mdd.time AND fd.asset_id = mdd.asset_id
            WHERE fd.time >= '{start_date}' AND a.is_active = TRUE
            ORDER BY fd.asset_id, fd.time ASC
        """
        try:
            return pl.read_database_uri(query, self.db_url)
        except Exception as e:
            self.logger.error(f"Failed to load alpha input data: {e}")
            return pl.DataFrame()
