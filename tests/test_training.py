import json
from pathlib import Path

from fraud_detection.training.train import TrainingResult


def test_reduced_training_flow_creates_complete_artifacts(
    trained_artifacts: tuple[Path, TrainingResult],
) -> None:
    artifacts_dir, result = trained_artifacts
    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))

    assert result.selected_model in {"logistic_regression", "random_forest", "catboost"}
    assert 0.0 <= result.threshold <= 1.0
    assert result.test_metrics.roc_auc >= 0.5
    assert metadata["schema_version"] == 1
    assert metadata["model_name"] == result.selected_model
    assert metadata["threshold"] == result.threshold
    assert metadata["model_selection"]["criterion"] == "mean_cv_roc_auc"
    assert set(metadata["candidate_cv_metrics"]) == {
        "logistic_regression",
        "random_forest",
        "catboost",
    }
    assert metadata["raw_feature_names"]
    assert metadata["feature_names"]
    assert metadata["categorical_features"] == ["city", "device"]
    assert (artifacts_dir / metadata["model_file"]).is_file()
