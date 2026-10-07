"""Command-line entry point for the training flow."""

import argparse
import logging
from pathlib import Path

from fraud_detection.config import N_SAMPLES
from fraud_detection.training.train import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud-detection model")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_training(
        TrainingConfig(
            n_samples=args.n_samples,
            artifacts_dir=args.artifacts_dir,
        )
    )


if __name__ == "__main__":
    main()
