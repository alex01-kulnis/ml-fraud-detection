"""Simple ML stages used by the training orchestrator."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraud_detection.config import (
    BASELINE_CATBOOST_DEPTH,
    BASELINE_CATBOOST_ITERATIONS,
    BASELINE_CATBOOST_LEARNING_RATE,
    N_SPLITS,
    RANDOM_STATE,
    RF_SEARCH_ITERATIONS,
    TARGET_COLUMN,
    CatBoostParameters,
)
from fraud_detection.data.dataset import DatasetSplit
from fraud_detection.evaluation.cross_validation import (
    CVResult,
    evaluate_catboost_model,
    evaluate_sklearn_model,
    generate_catboost_oof_probabilities,
    generate_sklearn_oof_probabilities,
)
from fraud_detection.evaluation.metrics import ClassificationMetrics, compute_metrics
from fraud_detection.evaluation.tuning import TuningResult, tune_catboost, tune_random_forest
from fraud_detection.features.engineering import add_features
from fraud_detection.models.factories import (
    build_catboost_model,
    build_logistic_model,
    build_random_forest_model,
    prepare_catboost_features,
)


@dataclass(frozen=True)
class SelectedModel:
    """The model choice and the CV evidence supporting it."""

    name: str
    params: dict[str, Any]
    cv_result: CVResult


def prepare_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target and apply the shared row-level feature engineering."""
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"dataset is missing target column: {TARGET_COLUMN}")
    return add_features(dataset.drop(columns=TARGET_COLUMN)), dataset[TARGET_COLUMN]


def evaluate_baselines(
    split: DatasetSplit,
    *,
    n_splits: int = N_SPLITS,
    n_jobs: int = -1,
    thread_count: int = -1,
) -> dict[str, CVResult]:
    """Evaluate the three baseline models using Train folds only."""
    baseline_catboost = CatBoostParameters(
        iterations=BASELINE_CATBOOST_ITERATIONS,
        learning_rate=BASELINE_CATBOOST_LEARNING_RATE,
        depth=BASELINE_CATBOOST_DEPTH,
    )
    return {
        "logistic_regression": evaluate_sklearn_model(
            build_logistic_model(),
            split.X_train,
            split.y_train,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
        ),
        "random_forest": evaluate_sklearn_model(
            build_random_forest_model(n_jobs=n_jobs),
            split.X_train,
            split.y_train,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
        ),
        "catboost": evaluate_catboost_model(
            split.X_train,
            split.y_train,
            baseline_catboost,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
            thread_count=thread_count,
        ),
    }


def tune_models(
    split: DatasetSplit,
    *,
    n_splits: int = N_SPLITS,
    rf_search_iterations: int = RF_SEARCH_ITERATIONS,
    n_jobs: int = -1,
    thread_count: int = -1,
) -> dict[str, TuningResult]:
    """Tune Random Forest and CatBoost on Train only."""
    return {
        "random_forest": tune_random_forest(
            split.X_train,
            split.y_train,
            n_iter=rf_search_iterations,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        ),
        "catboost": tune_catboost(
            split.X_train,
            split.y_train,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
            thread_count=thread_count,
        ),
    }


def select_best_model(
    baselines: Mapping[str, CVResult],
    tuned: Mapping[str, TuningResult],
) -> SelectedModel:
    """Choose the highest mean CV ROC-AUC after model-specific tuning."""
    candidates = (
        SelectedModel(
            name="logistic_regression",
            params={"max_iter": 1_000},
            cv_result=baselines["logistic_regression"],
        ),
        SelectedModel(
            name="random_forest",
            params=tuned["random_forest"].best_params,
            cv_result=tuned["random_forest"].cv_result,
        ),
        SelectedModel(
            name="catboost",
            params=tuned["catboost"].best_params,
            cv_result=tuned["catboost"].cv_result,
        ),
    )
    return max(candidates, key=lambda candidate: candidate.cv_result.roc_auc_mean)


def _build_model(
    selected: SelectedModel,
    *,
    n_jobs: int,
    thread_count: int,
) -> Any:
    if selected.name == "logistic_regression":
        return build_logistic_model()
    if selected.name == "random_forest":
        return build_random_forest_model(**selected.params, n_jobs=n_jobs)
    if selected.name == "catboost":
        return build_catboost_model(
            CatBoostParameters(**selected.params),
            thread_count=thread_count,
        )
    raise ValueError(f"unsupported model: {selected.name}")


def generate_oof_probabilities(
    selected: SelectedModel,
    split: DatasetSplit,
    *,
    n_splits: int = N_SPLITS,
    n_jobs: int = -1,
    thread_count: int = -1,
) -> NDArray[np.float64]:
    """Generate Train OOF probabilities for the selected model."""
    if selected.name == "catboost":
        return generate_catboost_oof_probabilities(
            split.X_train,
            split.y_train,
            CatBoostParameters(**selected.params),
            n_splits=n_splits,
            random_state=RANDOM_STATE,
            thread_count=thread_count,
        )
    model = _build_model(selected, n_jobs=n_jobs, thread_count=thread_count)
    return generate_sklearn_oof_probabilities(
        model,
        split.X_train,
        split.y_train,
        n_splits=n_splits,
        random_state=RANDOM_STATE,
    )


def fit_final_model(
    selected: SelectedModel,
    split: DatasetSplit,
    *,
    n_jobs: int = -1,
    thread_count: int = -1,
) -> Any:
    """Fit the selected model once on all Train rows."""
    model = _build_model(selected, n_jobs=n_jobs, thread_count=thread_count)
    train_features = split.X_train
    if selected.name == "catboost":
        train_features = prepare_catboost_features(train_features)
    model.fit(train_features, split.y_train)
    return model


def evaluate_final_model(
    model: Any,
    model_name: str,
    split: DatasetSplit,
    threshold: float,
) -> ClassificationMetrics:
    """Use the held-out Test partition exactly once for final metrics."""
    test_features = split.X_test
    if model_name == "catboost":
        test_features = prepare_catboost_features(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    return compute_metrics(split.y_test, probabilities, threshold=threshold)
