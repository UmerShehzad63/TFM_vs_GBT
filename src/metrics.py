"""
Evaluation metrics.
"""

import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test):

    start = time.perf_counter()

    predictions = model.predict(X_test)

    prediction_time = time.perf_counter() - start

    probabilities = model.predict_proba(X_test)[:, 1]

    return {

        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),

        "Prediction_Time": prediction_time,

    }