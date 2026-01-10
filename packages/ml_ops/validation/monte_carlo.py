import gc
import logging
import warnings
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from joblib import Parallel, delayed

from packages.ml_ops.validation.permutators.diagnostics import PermutationVerifier
from packages.ml_ops.protocols import ModelLifecycle, ValidationLogic


# --- Local Imports ---
from .base import BaseValidator, ValidationResult

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
    val_logic: ValidationLogic,
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
    try:
        permuted_df = val_logic.permute_data(raw_df_pandas, seed)
        if permuted_df.empty:
            return None

        # --- 2. RE-FEATURE ---
        factory = MLComponentFactory(settings, worker_logger)
        p_df = pl.from_pandas(permuted_df)
        feat_factory = FeatureFactory(settings)
        with_features = feat_factory.generate_all(p_df)

        # --- 3. RE-LABEL (THE CRITICAL FIX) ---
        try:
            dataset = val_logic.relabel_data(with_features, data_conf)
        except ValueError:
            return None

        # --- 4. PIPELINE & TRAIN ---
        from packages.ml_ops.modeling.pipeline import HorizonPipeline
        from packages.data_pipelines.processors.temporal import TemporalFeatureProcessor

        processors = []
        # Explicitly check the config dictionary or object
        gen_lags = (
            data_conf.generate_lags if hasattr(data_conf, "generate_lags") else False
        )

        if gen_lags:
            processors.append(TemporalFeatureProcessor())

        pipeline = HorizonPipeline(
            model=factory.create_model(model_conf),
            # Pass empty lists here; we will set trained_features directly
            feature_prefixes=[],
            target=data_conf.target_column,
            processors=processors,
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

        result = metrics.get(train_conf.eval_metric)

        # CLEANUP
        del (
            permuted_df,
            p_df,
            with_features,
            dataset,
            processed,
            X,
            y,
            strategy,
            evaluator,
        )
        gc.collect()  # Force release memory

        return result
    except Exception as e:
        # Log error locally if needed
        return None


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
        lifecycle: ModelLifecycle,
    ):
        super().__init__(logger)
        self.factory = factory
        self.data_conf = data_config
        self.train_conf = training_config
        self.val_conf = validation_config
        self.model_config = model_config
        self.lifecycle = lifecycle

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
            # Run the verifier
            # We pass ONLY the raw dataframe. The verifier will handle splitting/permuting for the check.
            verifier = PermutationVerifier(self.logger)
            is_valid = verifier.verify(raw_df_pandas, permuted_df=None, extra_cols=None)

            if not is_valid:
                self.logger.error(
                    "Halting Monte Carlo test due to failed permutation verification."
                )
                return ValidationResult(
                    "MonteCarlo",
                    False,
                    {"p_value": 1.0, "error": "Permutation failed verification"},
                )

        self.logger.info(f"Using frozen feature list for all {n_sims} simulations.")

        # Extract the pickle-safe logic object
        val_logic = self.lifecycle.get_validation_logic()

        # OPTIMIZATION 1: Limit Parallelism
        # If your machine has 32GB RAM and dataset is 2GB,
        # running 16 workers = 32GB. Boom.
        # Set a safe limit, e.g., 4 workers max for Alpha models.
        max_jobs = 4 if self.data_conf.kind == "alpha" else -1

        # Force GC before starting heavy parallel work
        gc.collect()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn.utils.parallel")

            random_scores = Parallel(
                n_jobs=max_jobs,  # <--- CONTROLLED CONCURRENCY
                backend="loky",  # Robust process isolation
                max_nbytes=None,  # Disable memory mapping limits
                mmap_mode="r",  # Try to share read-only memory if possible (requires numpy conversion usually)
            )(
                delayed(_execute_permutation)(
                    i,
                    raw_df_pandas,
                    self.data_conf,
                    self.train_conf,
                    self.model_config,
                    artifacts.feature_names,
                    val_logic,
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
