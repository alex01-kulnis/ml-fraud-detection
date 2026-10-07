from fraud_detection.evaluation.cross_validation import CVResult
from fraud_detection.evaluation.metrics import ClassificationMetrics, ThresholdResult
from fraud_detection.evaluation.tuning import TuningResult
from fraud_detection.training.artifacts import build_training_metadata
from fraud_detection.training.pipeline import SelectedModel, select_best_model


def cv_result(roc_auc: float) -> CVResult:
    return CVResult(
        roc_auc_mean=roc_auc,
        roc_auc_std=0.01,
        f1_mean=0.8,
        f1_std=0.02,
    )


def test_select_best_model_compares_tuned_and_baseline_cv_roc_auc() -> None:
    baselines = {
        "logistic_regression": cv_result(0.91),
        "random_forest": cv_result(0.82),
        "catboost": cv_result(0.85),
    }
    tuned = {
        "random_forest": TuningResult(
            model_name="random_forest",
            best_params={"n_estimators": 100},
            cv_result=cv_result(0.93),
        ),
        "catboost": TuningResult(
            model_name="catboost",
            best_params={"iterations": 200, "learning_rate": 0.05, "depth": 4},
            cv_result=cv_result(0.92),
        ),
    }

    selected = select_best_model(baselines, tuned)

    assert selected == SelectedModel(
        name="random_forest",
        params={"n_estimators": 100},
        cv_result=cv_result(0.93),
    )


def test_build_training_metadata_owns_serialization_schema() -> None:
    baselines = {
        "logistic_regression": cv_result(0.91),
        "random_forest": cv_result(0.82),
        "catboost": cv_result(0.85),
    }
    tuned = {
        "random_forest": TuningResult(
            model_name="random_forest",
            best_params={"n_estimators": 100},
            cv_result=cv_result(0.93),
        ),
        "catboost": TuningResult(
            model_name="catboost",
            best_params={"iterations": 200, "learning_rate": 0.05, "depth": 4},
            cv_result=cv_result(0.92),
        ),
    }
    selected = SelectedModel(
        name="random_forest",
        params={"n_estimators": 100},
        cv_result=cv_result(0.93),
    )
    threshold = ThresholdResult(threshold=0.4, precision=0.8, recall=0.9, f1=0.85)
    test_metrics = ClassificationMetrics(
        roc_auc=0.94,
        precision=0.81,
        recall=0.88,
        f1=0.84,
        confusion_matrix=((80, 5), (3, 12)),
    )

    metadata = build_training_metadata(
        selected=selected,
        baselines=baselines,
        tuned=tuned,
        threshold=threshold,
        test_metrics=test_metrics,
        n_samples=1_000,
    )

    assert metadata["model_selection"] == {
        "criterion": "mean_cv_roc_auc",
        "selected_model": "random_forest",
    }
    assert metadata["model_params"] == {"n_estimators": 100}
    assert metadata["candidate_cv_metrics"]["random_forest"]["roc_auc_mean"] == 0.93
    assert metadata["threshold"] == 0.4
    assert metadata["training_data"]["n_samples"] == 1_000
