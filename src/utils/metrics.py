"""Classification metrics shared by training and evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    """Accuracy, macro-F1 and weighted-F1 over the fixed label set 0..num_classes-1."""
    labels = list(range(num_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def confusion_counts(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Confusion matrix of counts: rows = true class, columns = predicted class."""
    idx = np.asarray(y_true, dtype=np.int64) * num_classes + np.asarray(y_pred, dtype=np.int64)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def bootstrap_ci(y_true, y_pred, num_classes: int, n_samples: int = 1000, seed: int = 0,
                 level: float = 0.95) -> dict:
    """Percentile bootstrap CI of accuracy and macro-F1 (images resampled with replacement).

    Macro-F1 per resample is 2TP / (2TP + FP + FN) averaged over all classes, which equals
    ``f1_score(average="macro", zero_division=0)`` over the fixed label set.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1 = np.empty(n_samples), np.empty(n_samples)
    for b in range(n_samples):
        i = rng.integers(0, n, size=n)
        cm = confusion_counts(y_true[i], y_pred[i], num_classes)
        tp = np.diag(cm).astype(float)
        denom = cm.sum(axis=0) + cm.sum(axis=1)          # (TP + FP) + (TP + FN)
        acc[b] = tp.sum() / n
        f1[b] = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean()
    q = [50 * (1 - level), 50 * (1 + level)]
    return {"accuracy": np.percentile(acc, q).tolist(), "macro_f1": np.percentile(f1, q).tolist()}
