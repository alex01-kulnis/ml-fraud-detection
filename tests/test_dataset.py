import pandas as pd

from fraud_detection.config import RAW_FEATURES, TARGET_COLUMN
from fraud_detection.data.dataset import generate_dataset, split_dataset
from fraud_detection.features.engineering import add_features


def test_generate_dataset_is_deterministic_and_preserves_contract() -> None:
    first = generate_dataset(n_samples=200, random_state=17)
    second = generate_dataset(n_samples=200, random_state=17)

    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == [*RAW_FEATURES, TARGET_COLUMN]
    assert first[TARGET_COLUMN].sum() == 30
    assert first["income"].isna().any()
    assert first["city"].isna().any()


def test_split_dataset_returns_stratified_disjoint_partitions() -> None:
    dataset = generate_dataset(n_samples=200, random_state=42)
    features = add_features(dataset.drop(columns=TARGET_COLUMN))
    split = split_dataset(
        features,
        dataset[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
    )

    assert split.X_train.shape == (160, 12)
    assert split.X_test.shape == (40, 12)
    assert split.y_train.shape == (160,)
    assert split.y_test.shape == (40,)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert split.y_train.mean() == split.y_test.mean() == 0.15
