from pathlib import Path

import pytest

from fraud_detection.training.train import TrainingConfig, TrainingResult, run_training


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, TrainingResult]:
    artifacts_dir = tmp_path_factory.mktemp("trained-artifacts")
    config = TrainingConfig(
        n_samples=180,
        n_splits=3,
        artifacts_dir=artifacts_dir,
        rf_search_iterations=2,
        n_jobs=1,
        thread_count=1,
    )
    return artifacts_dir, run_training(config)
