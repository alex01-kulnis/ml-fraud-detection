# Fraud Detection Project Design

## Goal

Refactor the supplied monolithic educational fraud-classification script into a small,
production-shaped Python project while preserving its data generation and ML methodology.

## Architecture

- `data` owns deterministic synthetic data and missing-value generation.
- `features` owns one pure, shared training/inference feature function.
- `preprocessing` and `models` build estimators without fitting data.
- `evaluation` owns CV, tuning, OOF probabilities, threshold selection, and metrics.
- `training` orchestrates one split, selection, final fit, final Test evaluation, and persistence.
- `inference` loads either sklearn or CatBoost artifacts and exposes a Python prediction API.

## Training flow

Generate raw data, validate it, derive features, split once, evaluate baselines, tune Random
Forest and CatBoost on Train only, select the highest mean CV ROC-AUC, make OOF predictions,
tune the F1 threshold, fit the selected model on full Train, evaluate Test once, and persist.

CatBoost CV/tuning/OOF remain manual to avoid the Python 3.14 sklearn clone incompatibility.
Sklearn preprocessing remains inside Pipelines so each CV fold fits its own preprocessing.

## Artifacts and inference

Sklearn estimators use `model.joblib`; CatBoost uses `model.cbm`. `metadata.json` contains the
model type, selected threshold, feature schema, parameters, CV/Test metrics, and schema version.
Inference validates raw columns, calls the same feature function, prepares CatBoost categories
when applicable, and returns probability, boolean class, and threshold.

## Verification

Unit tests cover feature purity and calculations, preprocessors, dataset/split contracts,
evaluation, artifact loading, and threshold application. A lightweight integration test runs
the complete training flow with reduced parameters. The normal training script is then run
with its full defaults and its artifacts exercised through inference.

