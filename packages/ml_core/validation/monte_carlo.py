import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging

from packages.ml_core.validation.base import BaseValidator, ValidationResult
from packages.ml_core.training.factory import MLComponentFactory
from packages.ml_core.common.schemas import (
    TrainingConfig,
    ValidationConfig,
    ModelConfig,
)
from packages.quant_lib.config import settings


# --- 1. THE WORKER FUNCTION (Must be outside the class) ---
def _execute_permutation(
    seed: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_conf: TrainingConfig,
    model_conf: ModelConfig,
):
    """
    Independent worker function.
    Instantiates its own Factory/Strategy to avoid pickling locks.
    """
    # Create a lightweight logger just for this worker process
    # We use standard logging here to avoid passing the complex LogManager structure
    worker_logger = logging.getLogger(f"mc_worker_{seed}")
    worker_logger.setLevel(logging.ERROR)  # Silence worker logs unless critical

    # Instantiate a FRESH Factory inside this process
    # Settings object is Pydantic, so it IS pickleable.
    factory = MLComponentFactory(settings, worker_logger)

    # 1. Shuffle Targets
    y_shuffled = y.sample(frac=1, random_state=seed).reset_index(drop=True)
    X_aligned = X.reset_index(drop=True)

    # 2. Fresh Model (DNA Clone)
    model_fresh = factory.create_model(model_conf)

    # 3. Internal Split (80/20)
    split = int(len(X_aligned) * 0.8)
    X_t, X_v = X_aligned.iloc[:split], X_aligned.iloc[split:]
    y_t, y_v = y_shuffled.iloc[:split], y_shuffled.iloc[split:]

    # 4. Train
    strategy = factory.create_strategy(train_conf)
    strategy.train(model_fresh, X_t, y_t, X_v, y_v)

    # 5. Score
    evaluator = factory.create_evaluator(train_conf)
    # Pass logger=None to keep stdout clean
    metrics = evaluator.evaluate(model_fresh, X_v, y_v, logger=None)

    return metrics.get(train_conf.eval_metric)


# --- 2. THE VALIDATOR CLASS ---
class MonteCarloValidator(BaseValidator):
    def __init__(
        self,
        logger,
        factory: MLComponentFactory,
        training_config: TrainingConfig,
        validation_config: ValidationConfig,
        model_config: ModelConfig,
    ):
        super().__init__(logger)
        self.factory = factory
        self.train_conf = training_config
        self.val_conf = validation_config
        self.model_config = model_config

    def validate(self, artifacts, tracker) -> ValidationResult:
        if not self.val_conf.monte_carlo_enabled:
            return ValidationResult("MonteCarlo", True, {"status": "skipped"})

        n_sims = self.val_conf.monte_carlo_simulations
        self.logger.info(f"Running Monte Carlo Permutation Test ({n_sims} runs)...")

        # Prepare Data
        X = artifacts.X_train
        y = artifacts.y_train

        # Execute Parallel Loop
        # We pass ONLY the data configs, not the 'self' instance
        random_scores = Parallel(n_jobs=-1)(
            delayed(_execute_permutation)(i, X, y, self.train_conf, self.model_config)
            for i in range(n_sims)
        )

        # Filter out failed runs (None)
        valid_scores = [s for s in random_scores if s is not None]

        # Calculate P-Value
        metric = self.train_conf.eval_metric
        real_score = artifacts.metrics.get(metric)
        if real_score is None:
            real_score = list(artifacts.metrics.values())[0]

        # Heuristic: Lower is Better for Loss/Error
        lower_is_better = metric in ["logloss", "mse", "rmse", "mae", "log_loss"]

        if lower_is_better:
            # Count how many random runs were BETTER (lower) than real
            better_randoms = sum(s <= real_score for s in valid_scores)
        else:
            # Count how many random runs were BETTER (higher) than real
            better_randoms = sum(s >= real_score for s in valid_scores)

        p_value = better_randoms / len(valid_scores) if valid_scores else 1.0

        tracker.log_metrics({"monte_carlo_p_value": p_value})

        # Save Distribution
        df_mc = pd.DataFrame({"random_scores": valid_scores})
        df_mc.to_csv("monte_carlo_dist.csv", index=False)
        tracker.log_artifact("monte_carlo_dist.csv")

        from pathlib import Path

        Path("monte_carlo_dist.csv").unlink()

        threshold = self.val_conf.monte_carlo_p_value_threshold
        passed = p_value < threshold

        msg = f"P-Value: {p_value:.4f} (Threshold: {threshold})"

        if passed:
            self.logger.success(f"✅ Monte Carlo Pass. {msg}")
        else:
            self.logger.error(f"❌ Monte Carlo Fail. {msg}")

        return ValidationResult("MonteCarlo", passed, {"p_value": p_value})


8
