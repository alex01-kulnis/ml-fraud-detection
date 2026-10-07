"""Train-only hyperparameter searches for Random Forest and CatBoost."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraud_detection.config import (
    CATBOOST_PARAM_CANDIDATES,
    N_SPLITS,
    RANDOM_STATE,
    RF_SEARCH_ITERATIONS,
    RF_SEARCH_SPACE,
    CatBoostParameters,
)
from fraud_detection.evaluation.cross_validation import (
    CVResult,
    evaluate_catboost_model,
    evaluate_sklearn_model,
)
from fraud_detection.models.factories import build_random_forest_model


@dataclass(frozen=True)
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    cv_result: CVResult


def tune_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    search_space: Mapping[str, Sequence[Any]] | None = None,
    n_iter: int = RF_SEARCH_ITERATIONS,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> TuningResult:
    """Tune a complete Random Forest Pipeline on Train only."""
    selected_space = search_space or dict(RF_SEARCH_SPACE)
    prefixed_space = {
        f"classifier__{parameter}": tuple(values) for parameter, values in selected_space.items()
    }
    search = RandomizedSearchCV(
        estimator=build_random_forest_model(random_state=random_state, n_jobs=1),
        param_distributions=prefixed_space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=False,
    )
    search.fit(features, target)
    best_params = {
        parameter.removeprefix("classifier__"): value
        for parameter, value in search.best_params_.items()
    }
    best_model = build_random_forest_model(
        **best_params,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    result = evaluate_sklearn_model(
        best_model,
        features,
        target,
        n_splits=n_splits,
        random_state=random_state,
    )
    return TuningResult(
        model_name="random_forest",
        best_params=best_params,
        cv_result=result,
    )


def tune_catboost(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    candidates: Sequence[CatBoostParameters] = CATBOOST_PARAM_CANDIDATES,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    thread_count: int = -1,
) -> TuningResult:
    """Tune CatBoost with explicit fold models, never sklearn.clone()."""
    if not candidates:
        raise ValueError("at least one CatBoost candidate is required")

    scored = [
        (
            parameters,
            evaluate_catboost_model(
                features,
                target,
                parameters,
                n_splits=n_splits,
                random_state=random_state,
                thread_count=thread_count,
            ),
        )
        for parameters in candidates
    ]
    best_parameters, best_cv = max(scored, key=lambda item: item[1].roc_auc_mean)
    return TuningResult(
        model_name="catboost",
        best_params=asdict(best_parameters),
        cv_result=best_cv,
    )
