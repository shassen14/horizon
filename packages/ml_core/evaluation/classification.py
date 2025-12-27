# packages/ml_core/evaluation/classification.py

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
        # .predict() gives the class (0 or 1)
        predictions = model.predict(X_test)
        # .predict_proba() gives the probability, needed for AUC
        # We take the probability of the "positive" class (class 1)
        probabilities = model.predict_proba(X_test)[:, 1]

        # 2. Calculate Key Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(
            f"Precision: {precision:.4f} (Of all 'Bull' predictions, how many were correct?)"
        )
        logger.info(
            f"Recall:    {recall:.4f} (Of all actual 'Bull' markets, how many did we catch?)"
        )
        logger.info(
            f"ROC AUC:   {roc_auc:.4f} (Overall model skill, 0.5 is random, 1.0 is perfect)"
        )

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
