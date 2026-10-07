"""Final classification metrics and OOF threshold selection."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from fraud_detection.config import THRESHOLDS


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    target: ArrayLike,
    probabilities: ArrayLike,
    *,
    threshold: float,
) -> ClassificationMetrics:
    """Compute probability and threshold-dependent binary metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    target_array = np.asarray(target)
    probability_array = np.asarray(probabilities, dtype=float)
    if target_array.shape != probability_array.shape:
        raise ValueError("target and probabilities must have the same shape")

    predictions = (probability_array >= threshold).astype(int)
    matrix = confusion_matrix(target_array, predictions, labels=[0, 1])
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target_array, probability_array)),
        precision=float(precision_score(target_array, predictions, zero_division=0)),
        recall=float(recall_score(target_array, predictions, zero_division=0)),
        f1=float(f1_score(target_array, predictions, zero_division=0)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )


def tune_threshold(
    target: ArrayLike,
    probabilities: ArrayLike,
    *,
    thresholds: Iterable[float] = THRESHOLDS,
    objective: Callable[[ThresholdResult], float] | None = None,
) -> ThresholdResult:
    """Select an OOF threshold; the scoring objective is replaceable."""
    target_array = np.asarray(target)
    probability_array = np.asarray(probabilities, dtype=float)
    if target_array.shape != probability_array.shape:
        raise ValueError("target and probabilities must have the same shape")

    candidates: list[ThresholdResult] = []
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("all thresholds must be between 0 and 1")
        predictions = (probability_array >= threshold).astype(int)
        candidates.append(
            ThresholdResult(
                threshold=float(threshold),
                precision=float(precision_score(target_array, predictions, zero_division=0)),
                recall=float(recall_score(target_array, predictions, zero_division=0)),
                f1=float(f1_score(target_array, predictions, zero_division=0)),
            )
        )
    if not candidates:
        raise ValueError("at least one threshold is required")

    score = objective or (lambda result: result.f1)
    return max(candidates, key=score)
