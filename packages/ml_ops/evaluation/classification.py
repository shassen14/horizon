# packages/ml_core/evaluation/classification.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
)

from packages.ml_ops.evaluation.base import EvaluationStrategy
from sklearn.exceptions import UndefinedMetricWarning
import warnings


class ClassificationEvaluator(EvaluationStrategy):
    """
    Evaluation strategy for classification models.
    Calculates Accuracy, Precision, Recall, AUC, LogLoss and shows a Confusion Matrix.
    """

    def evaluate(self, model, X_test, y_test, logger=None) -> dict:
        # 1. Prediction Phase
        predictions = model.predict(X_test)

        # Handle Probabilities (Needed for AUC and LogLoss)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        else:
            probs = None

        # 2. Setup Labels
        # We need the full universe of possible classes, even if some are not in the test set
        model_labels = getattr(model, "classes_", np.unique(y_test))
        test_labels = np.unique(y_test)
        all_labels = np.union1d(model_labels, test_labels)
        n_classes = len(all_labels)

        if logger:
            logger.info("--- Using ClassificationEvaluator ---")
            logger.info(
                f"Model trained on classes: {model_labels}. Test set has classes: {test_labels}."
            )

        # 3. Calculate Metrics
        # Defaults
        roc_auc = 0.5
        ll = 99.9  # Use a high penalty value if log_loss fails
        avg_method = "binary" if n_classes <= 2 else "weighted"

        # A. Probability-based metrics (AUC / LogLoss)
        if probs is not None:
            # ROC AUC
            try:
                with warnings.catch_warnings():
                    # Tell warnings to treat this specific one as an error so we can catch it
                    warnings.filterwarnings("error", category=UndefinedMetricWarning)
                    if n_classes <= 2:
                        roc_auc = roc_auc_score(y_test, probs[:, 1])
                    else:
                        roc_auc = roc_auc_score(
                            y_test,
                            probs,
                            multi_class="ovr",
                            average="weighted",
                            labels=all_labels,
                        )
            except (ValueError, UndefinedMetricWarning):  # Now we catch it
                # This happens if y_test contains only one class
                roc_auc = 0.5  # Default to random chance

            # Log Loss (Critical for Gradient Boosting evaluation)
            try:
                ll = log_loss(y_test, probs, labels=all_labels)
            except Exception:
                ll = 99.9  # Penalty for failure

        # B. Prediction-based metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average=avg_method, labels=all_labels, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average=avg_method, labels=all_labels, zero_division=0
        )

        # 4. Logging (Only if logger is provided)
        if logger:
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f} ({avg_method})")
            logger.info(f"  Recall:    {recall:.4f} ({avg_method})")
            logger.info(f"  ROC AUC:   {roc_auc:.4f} ({avg_method})")
            logger.info(f"  Log Loss:  {ll:.4f}")

            cm = confusion_matrix(y_test, predictions, labels=all_labels)
            logger.info(f"  Confusion Matrix (Labels: {all_labels}):\n{cm}")

            if roc_auc > 0.55:
                logger.success("  -> Model shows a predictive edge.")
            else:
                logger.warning("  -> Model performance is close to random chance.")

        # 5. Return Dictionary
        # We include logloss/log_loss so your config can pick either name
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

        if ll is not None:
            results["logloss"] = ll  # Matches LightGBM
            results["log_loss"] = ll  # Matches Scikit-Learn

        return results
