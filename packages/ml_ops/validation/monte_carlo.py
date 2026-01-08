import logging
import pandas as pd
import polars as pl
from pathlib import Path
from joblib import Parallel, delayed

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
):
    """
    An independent, parallel worker that runs one full permutation test:
    1. Creates a synthetic price history using OHLCPermutator.
    2. Recalculates all features on the synthetic data.
    3. Re-generates regime labels if necessary.
    4. Trains a fresh model.
    5. Evaluates and returns its score.
    """
    # Use a basic logger for workers to avoid passing complex objects
    worker_logger = logging.getLogger(f"mc_worker_{seed}")
    worker_logger.setLevel(logging.ERROR)

    factory = MLComponentFactory(settings, worker_logger)

    # --- 1. PERMUTE ---
    # Inject a dummy 'symbol' column if it's a single-asset (regime) dataset
    if MarketCol.SYMBOL not in raw_df_pandas.columns:
        raw_df_pandas = raw_df_pandas.copy()
        raw_df_pandas[MarketCol.SYMBOL] = "MarketContext"

    # Identify extra columns (like Breadth) to permute via arithmetic difference
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

    # --- 2. FEATURE ENGINEERING ---
    p_df = pl.from_pandas(permuted_df)
    feat_factory = FeatureFactory(settings)
    with_features = feat_factory.generate_all(p_df)

    # --- 3. LABELING (if Regime model) ---
    if data_conf.kind == "regime":
        from packages.data_pipelines.labeling.logic import RegimeLabeler

        price_col = "SPY" if "SPY" in with_features.columns else MarketCol.CLOSE
        labeler = RegimeLabeler(
            config=data_conf.labeling, horizon=data_conf.target_horizon_days
        )
        try:
            lab_df = labeler.fit_predict(with_features, price_col=price_col)
            dataset = with_features.join(lab_df, on=MarketCol.TIME, how="inner")
        except ValueError:
            return None  # Not enough data after permutation
    else:
        # For Alpha models, the target is calculated from the permuted price
        dataset = with_features.with_columns(
            (
                (
                    pl.col(MarketCol.CLOSE).shift(-data_conf.target_horizon_days)
                    / pl.col(MarketCol.CLOSE)
                )
                - 1
            )
            .over(MarketCol.ASSET_ID)  # Assuming asset_id exists for alpha
            .alias(data_conf.target_column)
        )

    # --- 4. PIPELINE & TRAIN ---
    from packages.ml_ops.modeling.pipeline import HorizonPipeline

    pipeline = HorizonPipeline(
        model=factory.create_model(model_conf),
        feature_prefixes=data_conf.feature_prefix_groups,
        target=data_conf.target_column,
        processors=[],  # Assuming no lag processors for this test to keep it simple
        exclude_patterns=data_conf.feature_exclude_patterns,
    )

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

        random_scores = Parallel(n_jobs=-1)(
            delayed(_execute_permutation)(
                i, raw_df_pandas, self.data_conf, self.train_conf, self.model_config
            )
            for i in range(n_sims)
        )

        valid_scores = [s for s in random_scores if s is not None and np.isfinite(s)]
        if not valid_scores:
            self.logger.error(
                "All Monte Carlo simulations failed to produce a valid score."
            )
            return ValidationResult(
                "MonteCarlo", False, {"p_value": 1.0, "error": "All runs failed"}
            )

        # Calculate P-Value
        metric = self.train_conf.eval_metric
        real_score = artifacts.metrics.get(metric)

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

        # Save Distribution for analysis
        df_mc = pd.DataFrame({"random_scores": valid_scores})
        df_mc.to_csv("monte_carlo_dist.csv", index=False)
        tracker.log_artifact("monte_carlo_dist.csv")
        Path("monte_carlo_dist.csv").unlink()

        threshold = self.val_conf.monte_carlo_p_value_threshold
        passed = p_value < threshold

        msg = f"P-Value: {p_value:.4f} (Threshold: {threshold}). Real score was beaten in {better_randoms}/{len(valid_scores)} random runs."

        if passed:
            self.logger.success(f"✅ Monte Carlo Pass. {msg}")
        else:
            self.logger.error(
                f"❌ Monte Carlo Fail. {msg} Model may be overfitting to spurious patterns."
            )

        return ValidationResult("MonteCarlo", passed, {"p_value": p_value})
