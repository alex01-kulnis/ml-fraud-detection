"""Linear orchestration of the fraud-detection training flow."""

import logging
from dataclasses import dataclass
from pathlib import Path

from fraud_detection.config import N_SAMPLES, N_SPLITS, RF_SEARCH_ITERATIONS, TARGET_COLUMN
from fraud_detection.data.dataset import generate_dataset, split_dataset
from fraud_detection.evaluation.cross_validation import CVResult
from fraud_detection.evaluation.metrics import ClassificationMetrics, tune_threshold
from fraud_detection.training.artifacts import ArtifactPaths, save_training_artifacts
from fraud_detection.training.pipeline import (
    evaluate_baselines,
    evaluate_final_model,
    fit_final_model,
    generate_oof_probabilities,
    prepare_features,
    select_best_model,
    tune_models,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Small set of useful runtime and smoke-test overrides."""

    n_samples: int = N_SAMPLES
    artifacts_dir: Path = Path("artifacts")
    n_splits: int = N_SPLITS
    rf_search_iterations: int = RF_SEARCH_ITERATIONS
    n_jobs: int = -1
    thread_count: int = -1


@dataclass(frozen=True)
class TrainingResult:
    selected_model: str
    threshold: float
    cv_metrics: CVResult
    test_metrics: ClassificationMetrics
    artifacts: ArtifactPaths


def run_training(config: TrainingConfig | None = None) -> TrainingResult:
    """Run model selection, threshold tuning, final evaluation, and persistence."""
    config = config or TrainingConfig()

    LOGGER.info("Generating dataset...")
    dataset = generate_dataset(n_samples=config.n_samples)
    features, target = prepare_features(dataset)
    split = split_dataset(features, target)
    LOGGER.info(
        "Dataset=%s Train=%d Test=%d target_rate=%.3f missing=%s",
        dataset.shape,
        len(split.X_train),
        len(split.X_test),
        dataset[TARGET_COLUMN].mean(),
        {name: int(value) for name, value in dataset.isna().sum().items() if value},
    )

    LOGGER.info("Evaluating baseline models...")
    baselines = evaluate_baselines(
        split,
        n_splits=config.n_splits,
        n_jobs=config.n_jobs,
        thread_count=config.thread_count,
    )
    for name, result in baselines.items():
        LOGGER.info(
            "%s baseline: ROC-AUC=%.4f F1=%.4f",
            name,
            result.roc_auc_mean,
            result.f1_mean,
        )

    LOGGER.info("Tuning Random Forest and CatBoost...")
    tuned = tune_models(
        split,
        n_splits=config.n_splits,
        rf_search_iterations=config.rf_search_iterations,
        n_jobs=config.n_jobs,
        thread_count=config.thread_count,
    )
    for name, result in tuned.items():
        LOGGER.info(
            "%s tuned: ROC-AUC=%.4f params=%s",
            name,
            result.cv_result.roc_auc_mean,
            result.best_params,
        )

    selected = select_best_model(baselines, tuned)
    LOGGER.info(
        "Selected model: %s; CV ROC-AUC=%.4f",
        selected.name,
        selected.cv_result.roc_auc_mean,
    )

    LOGGER.info("Generating OOF probabilities and tuning threshold...")
    oof_probabilities = generate_oof_probabilities(
        selected,
        split,
        n_splits=config.n_splits,
        n_jobs=config.n_jobs,
        thread_count=config.thread_count,
    )
    threshold = tune_threshold(split.y_train, oof_probabilities)
    LOGGER.info("Selected threshold: %.2f; OOF F1=%.4f", threshold.threshold, threshold.f1)

    LOGGER.info("Training final model on full Train...")
    final_model = fit_final_model(
        selected,
        split,
        n_jobs=config.n_jobs,
        thread_count=config.thread_count,
    )
    test_metrics = evaluate_final_model(
        final_model,
        selected.name,
        split,
        threshold.threshold,
    )
    LOGGER.info(
        "Final Test: ROC-AUC=%.4f Precision=%.4f Recall=%.4f F1=%.4f Confusion=%s",
        test_metrics.roc_auc,
        test_metrics.precision,
        test_metrics.recall,
        test_metrics.f1,
        test_metrics.confusion_matrix,
    )

    artifacts = save_training_artifacts(
        model=final_model,
        selected=selected,
        baselines=baselines,
        tuned=tuned,
        threshold=threshold,
        test_metrics=test_metrics,
        n_samples=config.n_samples,
        artifacts_dir=config.artifacts_dir,
    )
    LOGGER.info("Artifacts saved to %s", config.artifacts_dir)

    return TrainingResult(
        selected_model=selected.name,
        threshold=threshold.threshold,
        cv_metrics=selected.cv_result,
        test_metrics=test_metrics,
        artifacts=artifacts,
    )
