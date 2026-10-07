"""Synthetic data generation matching the original educational script."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import N_SAMPLES, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


@dataclass(frozen=True)
class DatasetSplit:
    """The project's single Train/Test partition."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def generate_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate the original synthetic fraud dataset without global RNG state."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    rng = np.random.RandomState(random_state)
    dataset = pd.DataFrame(
        {
            "age": rng.randint(18, 75, size=n_samples),
            "income": rng.normal(loc=5_000, scale=2_000, size=n_samples).clip(500),
            "amount": rng.exponential(scale=800, size=n_samples),
            "avg_amount_30d": rng.exponential(scale=500, size=n_samples) + 50,
            "transactions_24h": rng.poisson(lam=4, size=n_samples),
            "account_age_days": rng.randint(1, 3_000, size=n_samples),
            "transaction_hour": rng.randint(0, 24, size=n_samples),
            "city": rng.choice(["Berlin", "Warsaw", "Prague", "Vienna"], size=n_samples),
            "device": rng.choice(["iPhone", "Android", "Desktop"], size=n_samples),
            "is_new_device": rng.binomial(n=1, p=0.2, size=n_samples),
        }
    )

    fraud_score = (
        0.002 * dataset["amount"]
        + 0.25 * dataset["transactions_24h"]
        + 1.5 * dataset["is_new_device"]
        + 0.8 * (dataset["amount"] / dataset["avg_amount_30d"])
        + 1.2 * dataset["transaction_hour"].between(0, 4).astype(int)
        - 0.0004 * dataset["account_age_days"]
    )
    fraud_score += rng.normal(loc=0, scale=1.5, size=n_samples)
    fraud_threshold = np.percentile(fraud_score, 85)
    dataset[TARGET_COLUMN] = (fraud_score >= fraud_threshold).astype(int)

    dataset.loc[rng.random_sample(n_samples) < 0.08, "income"] = np.nan
    dataset.loc[rng.random_sample(n_samples) < 0.05, "city"] = np.nan
    return dataset


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Create the only stratified Train/Test split used by a training run."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
