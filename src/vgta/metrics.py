from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _f1(true: np.ndarray, predicted: np.ndarray, label: int) -> Dict[str, float]:
    true_positive = int(np.sum((true == label) & (predicted == label)))
    false_positive = int(np.sum((true != label) & (predicted == label)))
    false_negative = int(np.sum((true == label) & (predicted != label)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(
    truth: Iterable[int],
    predicted: Iterable[int],
    *,
    bloom_label: int,
    classes: int = 4,
) -> Dict[str, object]:
    true = np.asarray(list(truth), dtype=int)
    pred = np.asarray(list(predicted), dtype=int)
    if true.shape != pred.shape or true.ndim != 1 or true.size == 0:
        raise ValueError("classification arrays must be non-empty and equal-sized")
    if np.any((true < 0) | (true >= classes) | (pred < 0) | (pred >= classes)):
        raise ValueError("classification labels are outside the declared class range")
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (true, pred), 1)
    per_class: List[Dict[str, float]] = [_f1(true, pred, label) for label in range(classes)]
    bloom_true = (true == bloom_label).astype(int)
    bloom_pred = (pred == bloom_label).astype(int)
    bloom = _f1(bloom_true, bloom_pred, 1)
    return {
        "n": int(true.size),
        "accuracy": float(np.mean(true == pred)),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "bloom_f1": float(bloom["f1"]),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def reconstruction_metrics(
    truth: Iterable[float], predicted: Iterable[float], *, epsilon: float = 1e-8
) -> Dict[str, float]:
    true = np.asarray(list(truth), dtype=float)
    pred = np.asarray(list(predicted), dtype=float)
    if true.shape != pred.shape or true.ndim != 1 or true.size == 0:
        raise ValueError("reconstruction arrays must be non-empty and equal-sized")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("reconstruction metric inputs must be finite")
    error = pred - true
    absolute = np.abs(error)
    denominator = float(np.sum((true - true.mean()) ** 2))
    skill = 1.0 - float(np.sum(error ** 2)) / denominator if denominator > 0 else 1.0
    if np.std(true) == 0.0 or np.std(pred) == 0.0:
        r2 = 1.0 if np.allclose(true, pred) else 0.0
    else:
        r2 = float(np.corrcoef(true, pred)[0, 1] ** 2)
    return {
        "n": int(true.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(absolute / np.maximum(np.abs(true), epsilon))),
        "smape": float(
            100.0 * np.mean(2.0 * absolute / np.maximum(np.abs(true) + np.abs(pred), epsilon))
        ),
        "mdae": float(np.median(absolute)),
        "nse": skill,
        "r2": r2,
        "log_mae": float(
            np.mean(
                np.abs(
                    np.log(np.maximum(pred, epsilon)) - np.log(np.maximum(true, epsilon))
                )
            )
        ),
    }
