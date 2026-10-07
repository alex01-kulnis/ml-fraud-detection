from pathlib import Path

import numpy as np
import pytest

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    RAW_FEATURES,
    SCHEMA_VERSION,
    TARGET_COLUMN,
    CatBoostParameters,
)
from fraud_detection.data.dataset import generate_dataset
from fraud_detection.features.engineering import add_features
from fraud_detection.inference.predictor import FraudPredictor
from fraud_detection.models.factories import (
    build_catboost_model,
    build_logistic_model,
    prepare_catboost_features,
)
from fraud_detection.training.artifacts import save_model_artifacts
from fraud_detection.training.train import TrainingResult


def raw_transaction() -> dict[str, object]:
    dataset = generate_dataset(n_samples=20, random_state=31)
    return dataset.drop(columns=TARGET_COLUMN).iloc[0].to_dict()


def base_metadata() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "threshold": 0.5,
        "raw_feature_names": list(RAW_FEATURES),
        "feature_names": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "model_params": {},
        "cv_metrics": {},
        "test_metrics": {},
    }


def test_predictor_applies_saved_threshold_and_accepts_missing_category(
    trained_artifacts: tuple[Path, TrainingResult],
) -> None:
    artifacts_dir, _ = trained_artifacts
    predictor = FraudPredictor.load(artifacts_dir)
    transaction = raw_transaction()
    transaction["city"] = np.nan

    prediction = predictor.predict(transaction)

    assert 0.0 <= prediction["fraud_probability"] <= 1.0
    assert prediction["is_fraud"] == (prediction["fraud_probability"] >= prediction["threshold"])
    assert prediction["threshold"] == predictor.threshold


def test_predictor_rejects_missing_raw_feature(
    trained_artifacts: tuple[Path, TrainingResult],
) -> None:
    artifacts_dir, _ = trained_artifacts
    predictor = FraudPredictor.load(artifacts_dir)
    transaction = raw_transaction()
    transaction.pop("amount")

    with pytest.raises(ValueError, match="amount"):
        predictor.predict(transaction)


@pytest.mark.parametrize("model_name", ["logistic_regression", "catboost"])
def test_predictor_round_trips_both_model_formats(
    tmp_path: Path,
    model_name: str,
) -> None:
    dataset = generate_dataset(n_samples=80, random_state=21)
    features = add_features(dataset.drop(columns=TARGET_COLUMN))
    target = dataset[TARGET_COLUMN]
    if model_name == "catboost":
        model = build_catboost_model(
            CatBoostParameters(iterations=5, learning_rate=0.1, depth=2),
            random_state=21,
            thread_count=1,
        )
        model.fit(prepare_catboost_features(features), target)
    else:
        model = build_logistic_model(random_state=21)
        model.fit(features, target)

    artifact_dir = tmp_path / model_name
    save_model_artifacts(model, model_name, base_metadata(), artifact_dir)
    prediction = FraudPredictor.load(artifact_dir).predict(raw_transaction())

    assert 0.0 <= prediction["fraud_probability"] <= 1.0
