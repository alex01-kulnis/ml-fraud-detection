"""Pure row-level feature engineering shared by training and inference."""

import pandas as pd

from fraud_detection.config import RAW_FEATURES


def validate_raw_features(data: pd.DataFrame) -> None:
    """Validate the minimum raw schema accepted by training and inference."""
    if data.empty:
        raise ValueError("raw data must contain at least one row")
    missing = [feature for feature in RAW_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"missing required raw features: {', '.join(missing)}")


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the two derived fraud features."""
    validate_raw_features(data)
    featured = data.copy()
    featured["amount_ratio"] = featured["amount"] / featured["avg_amount_30d"]
    featured["is_night"] = featured["transaction_hour"].between(0, 4).astype(int)
    return featured
