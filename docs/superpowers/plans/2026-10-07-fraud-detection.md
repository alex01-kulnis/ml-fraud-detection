# Fraud Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, tested training and inference package from the supplied script.

**Architecture:** Focused modules expose typed functions and dataclasses. `training.train` is a
thin orchestration layer; sklearn pipelines and explicit CatBoost functions share structured
evaluation results and a common artifact metadata contract.

**Tech Stack:** Python 3.14, NumPy, pandas, scikit-learn, CatBoost, joblib, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-07-fraud-detection-design.md`

## Global Constraints

- Preserve the synthetic fraud target, missing-value rules, and two derived features.
- Split once with stratification and random state 42; Test is used only for final evaluation.
- Keep sklearn preprocessing inside Pipeline and CatBoost categoricals native.
- Use manual CatBoost CV/tuning/OOF; do not invoke sklearn clone-based utilities for it.
- Do not add services or infrastructure outside training, artifacts, inference, and tests.

## Review Focus

- A raw inference row with a missing categorical value still produces a probability.
- Missing required raw feature names fail before model prediction with a useful error.
- OOF probabilities cover each Train row exactly once and never use an in-sample model.
- Final model selection compares tuned candidates by the same CV ROC-AUC definition.
- Both sklearn and CatBoost artifact formats round-trip through the same predictor API.

---

### Task 1: Data and feature contracts

**Files:** `src/fraud_detection/config.py`, `src/fraud_detection/data/dataset.py`,
`src/fraud_detection/features/engineering.py`, `tests/test_dataset.py`,
`tests/test_features.py`

**Interfaces:** Produce `generate_dataset`, `split_dataset`, `validate_raw_features`, and
`add_features` for all later tasks.

- [x] Write tests for deterministic data, stratified contracts, feature values, and purity.
- [x] Run them and confirm missing-module failures.
- [x] Implement the minimal typed modules and run the full suite green.

### Task 2: Estimator construction

**Files:** `src/fraud_detection/preprocessing/builders.py`,
`src/fraud_detection/models/factories.py`, `tests/test_models.py`

**Interfaces:** Consume central feature lists; produce fresh Logistic, Random Forest, and
CatBoost estimators plus `prepare_catboost_features`.

- [x] Write tests for imputation/scaling/encoding, unknown categories, and fresh estimators.
- [x] Verify RED, implement builders/factories, and verify GREEN.

### Task 3: Evaluation and selection

**Files:** `src/fraud_detection/evaluation/metrics.py`,
`src/fraud_detection/evaluation/cross_validation.py`,
`src/fraud_detection/evaluation/tuning.py`, `tests/test_evaluation.py`

**Interfaces:** Produce structured metrics/CV/tuning results, generic OOF probabilities,
threshold tuning, and comparison inputs for training.

- [x] Test literal metric values, threshold choice, sklearn/CatBoost CV, tuning, and OOF shape.
- [x] Verify RED, implement one responsibility per function, and verify GREEN.

### Task 4: Training, persistence, and inference

**Files:** `src/fraud_detection/training/train.py`,
`src/fraud_detection/inference/predictor.py`, `tests/test_training.py`,
`tests/test_inference.py`

**Interfaces:** Consume Tasks 1-3; produce `TrainingConfig`, `TrainingResult`, `run_training`,
artifact files, and `FraudPredictor.predict`.

- [x] Test a reduced end-to-end run and artifact round trip without mocks.
- [x] Verify RED, implement orchestration/persistence/inference, and verify GREEN.

### Task 5: User entry points and full verification

**Files:** `scripts/train.py`, `README.md`, `artifacts/.gitkeep`

- [x] Add documented CLI usage and a raw-row inference example.
- [x] Run Ruff and the complete pytest suite.
- [x] Run default training, inspect metadata, and load the artifact for a real prediction.
