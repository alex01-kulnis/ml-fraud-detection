"""Stable model and metadata persistence."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    RANDOM_STATE,
    RAW_FEATURES,
    SCHEMA_VERSION,
    TARGET_COLUMN,
    TEST_SIZE,
)
from fraud_detection.evaluation.cross_validation import CVResult
from fraud_detection.evaluation.metrics import ClassificationMetrics, ThresholdResult
from fraud_detection.evaluation.tuning import TuningResult
from fraud_detection.training.pipeline import SelectedModel


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: Path
    metadata_path: Path


def build_training_metadata(
    *,
    selected: SelectedModel,
    baselines: dict[str, CVResult],
    tuned: dict[str, TuningResult],
    threshold: ThresholdResult,
    test_metrics: ClassificationMetrics,
    n_samples: int,
) -> dict[str, Any]:
    """Build the complete versioned metadata document in one place."""
    candidates = {
        "logistic_regression": baselines["logistic_regression"],
        "random_forest": tuned["random_forest"].cv_result,
        "catboost": tuned["catboost"].cv_result,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "threshold": threshold.threshold,
        "target_column": TARGET_COLUMN,
        "raw_feature_names": list(RAW_FEATURES),
        "feature_names": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "model_params": selected.params,
        "cv_metrics": selected.cv_result.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "threshold_metrics": threshold.to_dict(),
        "baseline_cv_metrics": {name: result.to_dict() for name, result in baselines.items()},
        "candidate_cv_metrics": {name: result.to_dict() for name, result in candidates.items()},
        "model_selection": {
            "criterion": "mean_cv_roc_auc",
            "selected_model": selected.name,
        },
        "training_data": {
            "n_samples": n_samples,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def save_model_artifacts(
    model: Any,
    model_name: str,
    metadata: dict[str, Any],
    artifacts_dir: Path,
) -> ArtifactPaths:
    """Persist a native CatBoost model or a complete sklearn Pipeline."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        if not isinstance(model, CatBoostClassifier):
            raise TypeError("catboost artifacts require CatBoostClassifier")
        model_file = "model.cbm"
        model_type = "CatBoostClassifier"
        model_path = artifacts_dir / model_file
        model.save_model(model_path)
    elif model_name in {"logistic_regression", "random_forest"}:
        model_file = "model.joblib"
        model_type = "LogisticRegression"
        if model_name == "random_forest":
            model_type = "RandomForestClassifier"
        model_path = artifacts_dir / model_file
        joblib.dump(model, model_path)
    else:
        raise ValueError(f"unsupported model_name: {model_name}")

    saved_metadata = {
        **metadata,
        "model_name": model_name,
        "model_type": model_type,
        "model_file": model_file,
    }
    metadata_path = artifacts_dir / "metadata.json"
    temporary_path = artifacts_dir / "metadata.json.tmp"
    temporary_path.write_text(
        json.dumps(saved_metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    return ArtifactPaths(model_path=model_path, metadata_path=metadata_path)


def save_training_artifacts(
    model: Any,
    selected: SelectedModel,
    baselines: dict[str, CVResult],
    tuned: dict[str, TuningResult],
    threshold: ThresholdResult,
    test_metrics: ClassificationMetrics,
    n_samples: int,
    artifacts_dir: Path,
) -> ArtifactPaths:
    """Build metadata and persist the final training artifacts."""
    metadata = build_training_metadata(
        selected=selected,
        baselines=baselines,
        tuned=tuned,
        threshold=threshold,
        test_metrics=test_metrics,
        n_samples=n_samples,
    )
    return save_model_artifacts(model, selected.name, metadata, artifacts_dir)
