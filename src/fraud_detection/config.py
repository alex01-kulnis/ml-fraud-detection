"""Central, immutable defaults shared by training and inference."""

from dataclasses import dataclass

RANDOM_STATE = 42
N_SAMPLES = 3_000
TEST_SIZE = 0.2
N_SPLITS = 5
TARGET_COLUMN = "fraud"
SCHEMA_VERSION = 1

RAW_NUMERIC_FEATURES = (
    "age",
    "income",
    "amount",
    "avg_amount_30d",
    "transactions_24h",
    "account_age_days",
    "transaction_hour",
    "is_new_device",
)
DERIVED_NUMERIC_FEATURES = ("amount_ratio", "is_night")
NUMERIC_FEATURES = (*RAW_NUMERIC_FEATURES, *DERIVED_NUMERIC_FEATURES)
CATEGORICAL_FEATURES = ("city", "device")
RAW_FEATURES = (
    "age",
    "income",
    "amount",
    "avg_amount_30d",
    "transactions_24h",
    "account_age_days",
    "transaction_hour",
    "city",
    "device",
    "is_new_device",
)
MODEL_FEATURES = (*RAW_FEATURES, *DERIVED_NUMERIC_FEATURES)

BASELINE_RF_ESTIMATORS = 200
BASELINE_CATBOOST_ITERATIONS = 300
BASELINE_CATBOOST_LEARNING_RATE = 0.05
BASELINE_CATBOOST_DEPTH = 6

RF_SEARCH_SPACE = (
    ("n_estimators", (100, 200, 400, 600)),
    ("max_depth", (None, 5, 10, 20)),
    ("min_samples_split", (2, 5, 10, 20)),
    ("min_samples_leaf", (1, 2, 5, 10)),
    ("max_features", ("sqrt", "log2", None)),
)
RF_SEARCH_ITERATIONS = 30


@dataclass(frozen=True)
class CatBoostParameters:
    """A clone-free CatBoost candidate configuration."""

    iterations: int
    learning_rate: float
    depth: int


CATBOOST_PARAM_CANDIDATES = (
    CatBoostParameters(iterations=200, learning_rate=0.05, depth=4),
    CatBoostParameters(iterations=300, learning_rate=0.05, depth=6),
    CatBoostParameters(iterations=500, learning_rate=0.03, depth=6),
    CatBoostParameters(iterations=300, learning_rate=0.1, depth=6),
    CatBoostParameters(iterations=500, learning_rate=0.05, depth=8),
)

THRESHOLDS = tuple(round(value / 100, 2) for value in range(10, 91, 5))
