"""Load saved artifacts and score one raw transaction."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from fraud_detection.config import SCHEMA_VERSION
from fraud_detection.features.engineering import add_features
from fraud_detection.models.factories import prepare_catboost_features


class FraudPredictor:
    """Production-shaped local inference over persisted training artifacts."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self._model = model
        self._metadata = metadata
        self.threshold = float(metadata["threshold"])

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> FraudPredictor:
        """Load and validate model metadata plus the matching model format."""
        directory = Path(artifacts_dir)
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "threshold",
            "model_name",
            "model_file",
            "raw_feature_names",
            "feature_names",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(f"metadata is missing fields: {', '.join(missing)}")
        if metadata["schema_version"] != SCHEMA_VERSION:
            actual_version = metadata["schema_version"]
            raise ValueError(
                f"unsupported schema_version {actual_version}; expected {SCHEMA_VERSION}"
            )
        threshold = float(metadata["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("metadata threshold must be between 0 and 1")

        model_path = directory / metadata["model_file"]
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        if metadata["model_name"] == "catboost":
            model = CatBoostClassifier()
            model.load_model(model_path)
        elif metadata["model_name"] in {"logistic_regression", "random_forest"}:
            model = joblib.load(model_path)
        else:
            raise ValueError(f"unsupported model_name: {metadata['model_name']}")
        return cls(model=model, metadata=metadata)

    def predict(self, transaction: Mapping[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Score one raw transaction and apply the saved decision threshold."""
        if isinstance(transaction, pd.DataFrame):
            raw = transaction.copy()
        else:
            raw = pd.DataFrame([dict(transaction)])
        if len(raw) != 1:
            raise ValueError("predict expects exactly one transaction")

        expected_raw = list(self._metadata["raw_feature_names"])
        missing = [column for column in expected_raw if column not in raw.columns]
        if missing:
            raise ValueError(f"missing required raw features: {', '.join(missing)}")
        featured = add_features(raw.loc[:, expected_raw])
        model_features = featured.loc[:, list(self._metadata["feature_names"])]
        if self._metadata["model_name"] == "catboost":
            model_features = prepare_catboost_features(model_features)

        probability = float(self._model.predict_proba(model_features)[0, 1])
        return {
            "fraud_probability": probability,
            "is_fraud": probability >= self.threshold,
            "threshold": self.threshold,
        }
