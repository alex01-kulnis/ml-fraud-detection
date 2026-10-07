"""Structured CV and OOF paths for sklearn and clone-incompatible CatBoost."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraud_detection.config import N_SPLITS, RANDOM_STATE, CatBoostParameters
from fraud_detection.models.factories import build_catboost_model, prepare_catboost_features


@dataclass(frozen=True)
class CVResult:
    roc_auc_mean: float
    roc_auc_std: float
    f1_mean: float
    f1_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _cv_result(roc_auc_scores: list[float], f1_scores: list[float]) -> CVResult:
    return CVResult(
        roc_auc_mean=float(np.mean(roc_auc_scores)),
        roc_auc_std=float(np.std(roc_auc_scores)),
        f1_mean=float(np.mean(f1_scores)),
        f1_std=float(np.std(f1_scores)),
    )


def _splitter(n_splits: int, random_state: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_sklearn_model(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> CVResult:
    """Evaluate a complete sklearn Pipeline without preprocessing leakage."""
    scores = cross_validate(
        estimator,
        features,
        target,
        cv=_splitter(n_splits, random_state),
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
    )
    return _cv_result(
        scores["test_roc_auc"].tolist(),
        scores["test_f1"].tolist(),
    )


def evaluate_catboost_model(
    features: pd.DataFrame,
    target: pd.Series,
    parameters: CatBoostParameters,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    thread_count: int = -1,
) -> CVResult:
    """Evaluate CatBoost manually, avoiding sklearn.clone()."""
    roc_auc_scores: list[float] = []
    f1_scores: list[float] = []
    for train_indices, validation_indices in _splitter(n_splits, random_state).split(
        features, target
    ):
        fold_train = prepare_catboost_features(features.iloc[train_indices])
        fold_validation = prepare_catboost_features(features.iloc[validation_indices])
        fold_target = target.iloc[train_indices]
        validation_target = target.iloc[validation_indices]
        model = build_catboost_model(
            parameters,
            random_state=random_state,
            thread_count=thread_count,
        )
        model.fit(fold_train, fold_target)
        probabilities = model.predict_proba(fold_validation)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        roc_auc_scores.append(float(roc_auc_score(validation_target, probabilities)))
        f1_scores.append(float(f1_score(validation_target, predictions, zero_division=0)))
    return _cv_result(roc_auc_scores, f1_scores)


def generate_sklearn_oof_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> NDArray[np.float64]:
    """Predict every row with an sklearn model that did not fit that row."""
    probabilities = np.full(len(features), np.nan, dtype=float)
    for train_indices, validation_indices in _splitter(n_splits, random_state).split(
        features, target
    ):
        model = clone(estimator)
        model.fit(features.iloc[train_indices], target.iloc[train_indices])
        probabilities[validation_indices] = model.predict_proba(features.iloc[validation_indices])[
            :, 1
        ]
    if np.isnan(probabilities).any():
        raise RuntimeError("OOF generation did not predict every row")
    return probabilities


def generate_catboost_oof_probabilities(
    features: pd.DataFrame,
    target: pd.Series,
    parameters: CatBoostParameters,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    thread_count: int = -1,
) -> NDArray[np.float64]:
    """Predict every row with a manually-created CatBoost fold model."""
    probabilities = np.full(len(features), np.nan, dtype=float)
    for train_indices, validation_indices in _splitter(n_splits, random_state).split(
        features, target
    ):
        fold_train = prepare_catboost_features(features.iloc[train_indices])
        fold_validation = prepare_catboost_features(features.iloc[validation_indices])
        model = build_catboost_model(
            parameters,
            random_state=random_state,
            thread_count=thread_count,
        )
        model.fit(fold_train, target.iloc[train_indices])
        probabilities[validation_indices] = model.predict_proba(fold_validation)[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("OOF generation did not predict every row")
    return probabilities
