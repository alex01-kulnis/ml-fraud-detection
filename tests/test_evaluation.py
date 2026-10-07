import numpy as np
import pytest

from fraud_detection.config import TARGET_COLUMN, CatBoostParameters
from fraud_detection.data.dataset import generate_dataset
from fraud_detection.evaluation.cross_validation import (
    evaluate_catboost_model,
    evaluate_sklearn_model,
    generate_catboost_oof_probabilities,
    generate_sklearn_oof_probabilities,
)
from fraud_detection.evaluation.metrics import compute_metrics, tune_threshold
from fraud_detection.evaluation.tuning import tune_catboost, tune_random_forest
from fraud_detection.features.engineering import add_features
from fraud_detection.models.factories import build_logistic_model


def make_evaluation_data(n_samples: int = 120):
    dataset = generate_dataset(n_samples=n_samples, random_state=9)
    features = add_features(dataset.drop(columns=TARGET_COLUMN))
    return features, dataset[TARGET_COLUMN]


def test_compute_metrics_uses_probabilities_and_selected_threshold() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = compute_metrics(target, probabilities, threshold=0.5)

    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.confusion_matrix == ((2, 0), (1, 1))


def test_tune_threshold_maximizes_oof_f1() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    result = tune_threshold(target, probabilities, thresholds=(0.3, 0.5, 0.7))

    assert result.threshold == pytest.approx(0.3)
    assert result.f1 == pytest.approx(0.8)
    assert result.recall == pytest.approx(1.0)


def test_sklearn_cv_and_oof_return_complete_probabilities() -> None:
    features, target = make_evaluation_data()
    model = build_logistic_model(random_state=11)

    result = evaluate_sklearn_model(model, features, target, n_splits=3, random_state=11)
    oof = generate_sklearn_oof_probabilities(
        model,
        features,
        target,
        n_splits=3,
        random_state=11,
    )

    assert 0.5 <= result.roc_auc_mean <= 1.0
    assert result.roc_auc_std >= 0.0
    assert 0.0 <= result.f1_mean <= 1.0
    assert oof.shape == (len(features),)
    assert np.isfinite(oof).all()
    assert ((0.0 <= oof) & (oof <= 1.0)).all()


def test_manual_catboost_cv_and_oof_return_complete_probabilities() -> None:
    features, target = make_evaluation_data()
    params = CatBoostParameters(iterations=8, learning_rate=0.1, depth=2)

    result = evaluate_catboost_model(
        features,
        target,
        params,
        n_splits=3,
        random_state=11,
        thread_count=1,
    )
    oof = generate_catboost_oof_probabilities(
        features,
        target,
        params,
        n_splits=3,
        random_state=11,
        thread_count=1,
    )

    assert 0.5 <= result.roc_auc_mean <= 1.0
    assert 0.0 <= result.f1_mean <= 1.0
    assert oof.shape == (len(features),)
    assert np.isfinite(oof).all()


def test_tuners_return_winning_parameters_and_cv_metrics() -> None:
    features, target = make_evaluation_data()
    rf_result = tune_random_forest(
        features,
        target,
        search_space={
            "n_estimators": (5, 10),
            "max_depth": (2, None),
            "min_samples_split": (2,),
            "min_samples_leaf": (1,),
            "max_features": ("sqrt",),
        },
        n_iter=2,
        n_splits=3,
        random_state=11,
        n_jobs=1,
    )
    cat_result = tune_catboost(
        features,
        target,
        candidates=(
            CatBoostParameters(iterations=5, learning_rate=0.1, depth=2),
            CatBoostParameters(iterations=8, learning_rate=0.1, depth=3),
        ),
        n_splits=3,
        random_state=11,
        thread_count=1,
    )

    assert rf_result.model_name == "random_forest"
    assert rf_result.best_params["n_estimators"] in {5, 10}
    assert 0.5 <= rf_result.cv_result.roc_auc_mean <= 1.0
    assert cat_result.model_name == "catboost"
    assert cat_result.best_params["iterations"] in {5, 8}
    assert 0.5 <= cat_result.cv_result.roc_auc_mean <= 1.0
