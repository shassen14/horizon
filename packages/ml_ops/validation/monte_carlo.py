import logging
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from joblib import Parallel, delayed

from packages.data_pipelines.labeling.logic import RegimeLabeler
from packages.ml_ops.validation.permutators.diagnostics import PermutationVerifier

# --- Local Imports ---
from .base import BaseValidator, ValidationResult
from .permutators.ohlc import OHLCPermutator

# --- Library Imports ---
from packages.ml_ops.training.factory import MLComponentFactory
from packages.quant_lib.config import settings
from packages.quant_lib.features import FeatureFactory

# --- Contract Imports ---
from packages.contracts.blueprints import (
    TrainingConfig,
    ValidationConfig,
    ModelConfig,
    DataConfigType,
)
from packages.contracts.vocabulary.columns import MarketCol

# --------------------------------------------------------------------------------------
# WORKER FUNCTION (Defined globally to be pickle-safe for multiprocessing)
# --------------------------------------------------------------------------------------


def _execute_permutation(
    seed: int,
    raw_df_pandas: pd.DataFrame,
    data_conf: DataConfigType,
    train_conf: TrainingConfig,
    model_conf: ModelConfig,
    feature_list: list,
):
    """
    An independent, parallel worker that runs one full permutation test:
    1. Creates a synthetic price history using OHLCPermutator.
    2. Recalculates all features on the synthetic data.
    3. Re-generates regime labels if necessary.
    4. Trains a fresh model.
    5. Evaluates and returns its score.
    """
    worker_logger = logging.getLogger(f"mc_worker_{seed}")
    worker_logger.setLevel(logging.ERROR)

    factory = MLComponentFactory(settings, worker_logger)

    if data_conf.target_column in raw_df_pandas.columns:
        raw_df_pandas = raw_df_pandas.drop(columns=[data_conf.target_column])

    # --- 1. PERMUTE ---
    if MarketCol.SYMBOL not in raw_df_pandas.columns:
        raw_df_pandas = raw_df_pandas.copy()
        raw_df_pandas[MarketCol.SYMBOL] = "MarketContext"

    ohlc_cols = {
        MarketCol.OPEN,
        MarketCol.HIGH,
        MarketCol.LOW,
        MarketCol.CLOSE,
        MarketCol.VOLUME,
        MarketCol.SYMBOL,
        MarketCol.TIME,
    }
    extra_cols = [c for c in raw_df_pandas.columns if c not in ohlc_cols]

    permutator = OHLCPermutator(diff_cols=extra_cols)
    permuted_df = permutator.permute(raw_df_pandas, seed=seed)

    # --- 2. RE-FEATURE ---
    p_df = pl.from_pandas(permuted_df)
    feat_factory = FeatureFactory(settings)
    with_features = feat_factory.generate_all(p_df)

    # --- 3. RE-LABEL (THE CRITICAL FIX) ---
    if data_conf.kind == "regime":
        price_col = "SPY" if "SPY" in with_features.columns else MarketCol.CLOSE

        # Instantiate the pure labeling logic class
        labeler = RegimeLabeler(
            config=data_conf.labeling, horizon=data_conf.target_horizon_days
        )
        try:
            # Generate new labels based on the permuted price path
            lab_df = labeler.fit_predict(with_features, price_col=price_col)
            # Inner join to align features with the newly created labels
            dataset = with_features.join(lab_df, on=MarketCol.TIME, how="inner")
        except ValueError:
            return None  # Permutation created a sequence too short for labeling
    else:  # For Alpha models
        dataset = with_features.with_columns(
            (
                (
                    pl.col(MarketCol.CLOSE).shift(-data_conf.target_horizon_days)
                    / pl.col(MarketCol.CLOSE)
                )
                - 1
            )
            .over(MarketCol.ASSET_ID)
            .alias(data_conf.target_column)
        )

    # --- 4. PIPELINE & TRAIN ---
    from packages.ml_ops.modeling.pipeline import HorizonPipeline

    pipeline = HorizonPipeline(
        model=factory.create_model(model_conf),
        # Pass empty lists here; we will set trained_features directly
        feature_prefixes=[],
        target=data_conf.target_column,
        processors=[],
        exclude_patterns=[],
    )

    pipeline.trained_features = feature_list

    processed = pipeline.preprocess(dataset)
    X, y = pipeline.get_X_y(processed)

    if X is None or y is None or len(X) < 100:
        return None

    # Simple 80/20 split for this permutation run
    split_idx = int(len(X) * 0.8)
    X_t, X_v = X.iloc[:split_idx], X.iloc[split_idx:]
    y_t, y_v = y.iloc[:split_idx], y.iloc[split_idx:]

    strategy = factory.create_strategy(train_conf)
    strategy.train(pipeline.model, X_t, y_t, X_v, y_v)

    evaluator = factory.create_evaluator(train_conf)
    metrics = evaluator.evaluate(pipeline.model, X_v, y_v, logger=None)

    return metrics.get(train_conf.eval_metric)


# --------------------------------------------------------------------------------------
# VALIDATOR CLASS
# --------------------------------------------------------------------------------------


class MonteCarloValidator(BaseValidator):
    def __init__(
        self,
        logger,
        factory: MLComponentFactory,
        data_config: DataConfigType,
        training_config: TrainingConfig,
        validation_config: ValidationConfig,
        model_config: ModelConfig,
    ):
        super().__init__(logger)
        self.factory = factory
        self.data_conf = data_config
        self.train_conf = training_config
        self.val_conf = validation_config
        self.model_config = model_config

    def validate(self, artifacts, tracker) -> ValidationResult:
        if not self.val_conf.monte_carlo_enabled:
            return ValidationResult("MonteCarlo", True, {"status": "skipped"})

        n_sims = self.val_conf.monte_carlo_simulations
        self.logger.info(
            f"--- Running Monte Carlo Permutation Test ({n_sims} runs) ---"
        )

        raw_df_pandas = artifacts.raw_df.to_pandas()

        # Self Verification
        if n_sims > 0:
            # Prepare data for verification
            clean_df = raw_df_pandas.drop(
                columns=[self.data_conf.target_column], errors="ignore"
            )
            if "symbol" not in clean_df.columns:
                clean_df["symbol"] = "MarketContext"

            # Run the first permutation for verification
            ohlc_cols = {"open", "high", "low", "close", "volume", "symbol", "time"}
            extra_cols = [c for c in clean_df.columns if c not in ohlc_cols]

            permutator = OHLCPermutator(diff_cols=extra_cols)
            permuted_df_for_verify = permutator.permute(clean_df, seed=0)

            # Run the verifier
            verifier = PermutationVerifier(self.logger)
            is_valid = verifier.verify(clean_df, permuted_df_for_verify, extra_cols)

            # Optionally, halt the test if verification fails
            if not is_valid:
                self.logger.error(
                    "Halting Monte Carlo test due to failed permutation verification."
                )
                return ValidationResult(
                    "MonteCarlo",
                    False,
                    {"p_value": 1.0, "error": "Permutation failed verification"},
                )

        final_feature_names = artifacts.feature_names
        self.logger.info(f"Using frozen feature list for all {n_sims} simulations.")

        random_scores = Parallel(n_jobs=-1)(
            delayed(_execute_permutation)(
                i,
                raw_df_pandas,
                self.data_conf,
                self.train_conf,
                self.model_config,
                final_feature_names,
            )
            for i in range(n_sims)
        )

        valid_scores = [s for s in random_scores if s is not None and np.isfinite(s)]
        if not valid_scores:
            self.logger.error("All Monte Carlo simulations failed.")
            return ValidationResult(
                "MonteCarlo", False, {"p_value": 1.0, "error": "All runs failed"}
            )

        # Calculate P-Value
        metric = self.train_conf.eval_metric
        real_score = artifacts.metrics.get(metric, 0)

        lower_is_better = metric.lower() in [
            "logloss",
            "log_loss",
            "mse",
            "rmse",
            "mae",
        ]

        lower_is_better = metric.lower() in [
            "logloss",
            "log_loss",
            "mse",
            "rmse",
            "mae",
        ]
        if lower_is_better:
            better_randoms = sum(s <= real_score for s in valid_scores)
        else:
            better_randoms = sum(s >= real_score for s in valid_scores)

        p_value = better_randoms / len(valid_scores)

        tracker.log_metrics({"monte_carlo_p_value": p_value})

        df_mc = pd.DataFrame({"random_scores": valid_scores})
        df_mc.to_csv("monte_carlo_dist.csv", index=False)
        tracker.log_artifact("monte_carlo_dist.csv")
        Path("monte_carlo_dist.csv").unlink()

        threshold = self.val_conf.monte_carlo_p_value_threshold
        passed = p_value < threshold

        msg = f"P-Value: {p_value:.4f} (Threshold: {threshold})."

        if passed:
            self.logger.success(f"✅ Monte Carlo Pass. {msg}")
        else:
            self.logger.error(f"❌ Monte Carlo Fail. {msg}")

        return ValidationResult("MonteCarlo", passed, {"p_value": p_value})
