import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.engineering import add_features, validate_raw_features


def make_raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [1_000.0, np.nan, 3_000.0],
            "amount": [100.0, 90.0, 50.0],
            "avg_amount_30d": [25.0, 30.0, 100.0],
            "transactions_24h": [1, 2, 3],
            "account_age_days": [100, 200, 300],
            "transaction_hour": [0, 4, 5],
            "city": ["Berlin", np.nan, "Prague"],
            "device": ["iPhone", "Android", "Desktop"],
            "is_new_device": [0, 1, 0],
        }
    )


def test_add_features_calculates_values_without_mutating_input() -> None:
    raw = make_raw_rows()
    original = raw.copy(deep=True)

    result = add_features(raw)

    pd.testing.assert_frame_equal(raw, original)
    assert result["amount_ratio"].tolist() == [4.0, 3.0, 0.5]
    assert result["is_night"].tolist() == [1, 1, 0]


def test_validate_raw_features_lists_missing_columns() -> None:
    raw = make_raw_rows().drop(columns=["amount", "city"])

    with pytest.raises(ValueError, match="amount, city"):
        validate_raw_features(raw)


def test_validate_raw_features_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must contain at least one row"):
        validate_raw_features(make_raw_rows().iloc[0:0])
