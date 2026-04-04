import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted"),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted"),
    }


def print_report(y_true, y_pred) -> None:
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def aggregate_fold_metrics(fold_metrics: list) -> dict:
    keys = fold_metrics[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std": float(np.std([m[k] for m in fold_metrics])),
        }
        for k in keys
    }
