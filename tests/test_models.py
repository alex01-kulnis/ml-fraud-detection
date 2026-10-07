import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import CATEGORICAL_FEATURES, CatBoostParameters
from fraud_detection.features.engineering import add_features
from fraud_detection.models.factories import (
    build_catboost_model,
    build_logistic_model,
    build_random_forest_model,
    prepare_catboost_features,
)


def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "income": [1_000.0, np.nan, 2_000.0, 3_000.0, 4_000.0, 5_000.0],
            "amount": [10.0, 900.0, 20.0, 800.0, 30.0, 700.0],
            "avg_amount_30d": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "transactions_24h": [1, 9, 1, 8, 2, 7],
            "account_age_days": [900, 10, 800, 20, 700, 30],
            "transaction_hour": [12, 2, 13, 3, 14, 4],
            "city": ["Berlin", "Warsaw", np.nan, "Vienna", "Berlin", "Warsaw"],
            "device": ["Desktop", "Android", "Desktop", "iPhone", "Desktop", "Android"],
            "is_new_device": [0, 1, 0, 1, 0, 1],
        }
    )
    return add_features(raw), pd.Series([0, 1, 0, 1, 0, 1])


def test_logistic_pipeline_handles_missing_and_unknown_categories() -> None:
    features, target = make_training_data()
    model = build_logistic_model(random_state=7)
    model.fit(features, target)
    unseen = features.iloc[[0]].copy()
    unseen.loc[unseen.index[0], "city"] = "Tallinn"

    probability = model.predict_proba(unseen)[0, 1]

    assert isinstance(model, Pipeline)
    assert 0.0 <= probability <= 1.0
    numeric = model.named_steps["preprocessing"].named_transformers_["numeric"]
    assert isinstance(numeric.named_steps["scaler"], StandardScaler)


def test_random_forest_pipeline_uses_imputation_without_scaling() -> None:
    features, target = make_training_data()
    model = build_random_forest_model(n_estimators=5, random_state=7, n_jobs=1)
    model.fit(features, target)

    probability = model.predict_proba(features.iloc[[0]])[0, 1]

    numeric = model.named_steps["preprocessing"].named_transformers_["numeric"]
    assert 0.0 <= probability <= 1.0
    assert set(numeric.named_steps) == {"imputer"}


def test_catboost_preparation_is_pure_and_factory_returns_fresh_models() -> None:
    features, target = make_training_data()
    original = features.copy(deep=True)
    params = CatBoostParameters(iterations=3, learning_rate=0.1, depth=2)

    prepared = prepare_catboost_features(features)
    first = build_catboost_model(params, random_state=7, thread_count=1)
    second = build_catboost_model(params, random_state=7, thread_count=1)

    pd.testing.assert_frame_equal(features, original)
    assert prepared["city"].isna().sum() == 0
    assert "Unknown" in prepared["city"].tolist()
    assert all(
        prepared[column].map(lambda value: isinstance(value, str)).all()
        for column in CATEGORICAL_FEATURES
    )
    assert first is not second
    assert first.get_params()["cat_features"] == list(CATEGORICAL_FEATURES)
    first.fit(prepared, target)
    assert 0.0 <= first.predict_proba(prepared.iloc[[0]])[0, 1] <= 1.0
