"""Model construction and CatBoost-specific input preparation."""

from collections.abc import Sequence

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.config import (
    BASELINE_CATBOOST_DEPTH,
    BASELINE_CATBOOST_ITERATIONS,
    BASELINE_CATBOOST_LEARNING_RATE,
    BASELINE_RF_ESTIMATORS,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    CatBoostParameters,
)
from fraud_detection.preprocessing.builders import (
    build_logistic_preprocessor,
    build_random_forest_preprocessor,
)


def build_logistic_model(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build an unfitted Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_logistic_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=random_state),
            ),
        ]
    )


def build_random_forest_model(
    *,
    n_estimators: int = BASELINE_RF_ESTIMATORS,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """Build an unfitted Random Forest pipeline."""
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(
        steps=[
            ("preprocessing", build_random_forest_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_catboost_model(
    parameters: CatBoostParameters | None = None,
    *,
    random_state: int = RANDOM_STATE,
    thread_count: int = -1,
) -> CatBoostClassifier:
    """Build an unfitted CatBoost classifier with native categories."""
    selected = parameters or CatBoostParameters(
        iterations=BASELINE_CATBOOST_ITERATIONS,
        learning_rate=BASELINE_CATBOOST_LEARNING_RATE,
        depth=BASELINE_CATBOOST_DEPTH,
    )
    return CatBoostClassifier(
        iterations=selected.iterations,
        learning_rate=selected.learning_rate,
        depth=selected.depth,
        cat_features=list(CATEGORICAL_FEATURES),
        verbose=False,
        random_seed=random_state,
        thread_count=thread_count,
        allow_writing_files=False,
    )


def prepare_catboost_features(
    features: pd.DataFrame,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Return a CatBoost-safe copy with categorical missing values normalized."""
    missing = [column for column in categorical_features if column not in features.columns]
    if missing:
        raise ValueError(f"missing categorical features: {', '.join(missing)}")

    prepared = features.copy()
    for column in categorical_features:
        prepared[column] = prepared[column].fillna("Unknown").astype(str)
    return prepared
