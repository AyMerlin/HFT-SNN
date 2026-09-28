"""Three-class price-trend metrics (DeepLOB Tables I–II, plan §8.1).

The paper reports accuracy, precision, recall and F1, with precision/recall/F1 averaged over
the classes weighted by support (its recall equals its accuracy in every row). Macro
averages, Matthews' correlation and the confusion matrix are reported as well. Classes
without predictions or samples score 0 precision / recall, as in scikit-learn's
`zero_division=0`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Counts C[i, j] of samples with true class i predicted as j."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred differ in shape")
    if len(y_true) and (min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= n_classes):
        raise ValueError(f"class indices must lie in [0, {n_classes})")
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    precision, recall = _div(tp, predicted), _div(tp, support)
    f1 = _div(2 * precision * recall, precision + recall)
    weights = _div(support, np.full_like(support, n))
    # Matthews' correlation for K classes (Gorodkin 2004)
    cov_tp = tp.sum() * n - (support * predicted).sum()
    denom = np.sqrt((n**2 - (predicted**2).sum()) * (n**2 - (support**2).sum()))
    return {
        "n": int(n),
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "mcc": float(cov_tp / denom) if denom > 0 else 0.0,
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.astype(int).tolist(),
        },
        "confusion": cm.astype(int).tolist(),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred, n_classes))


SUMMARY_KEYS = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "mcc")


def mean_over_folds(results: Sequence[dict], keys: Sequence[str] = SUMMARY_KEYS) -> dict:
    """Unweighted mean of per-fold metrics (DeepLOB Setup 1: "mean … over all folds")."""
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
