# packages/ml_core/evaluation/classification.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from packages.ml_core.evaluation.base import EvaluationStrategy


class ClassificationEvaluator(EvaluationStrategy):
    """
    Evaluation strategy for classification models.
    Calculates Accuracy, Precision, Recall, AUC, and shows a Confusion Matrix.
    """

    def evaluate(self, model, X_test, y_test, logger) -> dict:
        logger.info("--- Using ClassificationEvaluator ---")

        # 1. Get Predictions
        predictions = model.predict(X_test)

        # 2. Get Unique Labels
        # We need to know all possible labels the model was trained on
        # and all labels that actually appear in our test set.
        model_labels = model.classes_
        test_labels = np.unique(y_test)

        # The full universe of labels for scoring
        all_labels = np.union1d(model_labels, test_labels)

        n_classes = len(all_labels)

        logger.info(
            f"Model trained on classes: {model_labels}. Test set has classes: {test_labels}."
        )

        # 3. Handle Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        else:
            probs = None

        # 4. Calculate Metrics (Dynamic)
        if n_classes <= 2:
            avg_method = "binary"
            roc_auc = 0.5
            if probs is not None:
                # Binary: prob of positive class (Class 1)
                roc_auc = roc_auc_score(y_test, probs[:, 1])
        else:
            # Multi-class
            avg_method = "weighted"
            roc_auc = 0.5
            if probs is not None:
                # --- THE FIX: Pass the 'labels' parameter ---
                # This tells roc_auc_score the full context of possible classes,
                # even if some are missing from y_test.
                roc_auc = roc_auc_score(
                    y_test,
                    probs,
                    multi_class="ovr",
                    average="weighted",
                    labels=all_labels,
                )

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average=avg_method, labels=all_labels, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average=avg_method, labels=all_labels, zero_division=0
        )

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f} (Weighted)")
        logger.info(f"Recall:    {recall:.4f} (Weighted)")
        logger.info(f"ROC AUC:   {roc_auc:.4f} (Weighted)")

        # 3. Display Confusion Matrix
        # [[True Negative, False Positive],
        #  [False Negative, True Positive]]
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"Confusion Matrix:\n{cm}")

        if roc_auc > 0.55:
            logger.success("✅ Model shows a predictive edge for classification.")
        else:
            logger.warning("❌ Model performance is close to random chance.")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }
