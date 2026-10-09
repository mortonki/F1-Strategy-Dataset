# Project Brief: F1 Strategy Prediction

## Goals
- Build a machine learning model to predict the next pit stop lap (`PitNextLap`) for F1 cars.
- Use historical race data to identify patterns in driver behavior and car degradation.
- Implement a robust hyperparameter tuning pipeline using Optuna.
- Track all experiments using MLflow for reproducibility and comparison.
- Provide a standalone inference script to load the best model and generate predictions.

## Scope
- Data preprocessing including lagged features and rolling averages.
- Model training using LightGBM and CatBoost.
- Evaluation metrics: Precision and AUC-ROC.
- Time-series aware data splitting (Train: 2022-2023, Val: 2024, Test: 2025).
- Three execution modes: `tune` (Optuna), `train` (fit with saved params), `evaluate` (load best model).
- Config-driven setup via `config.yaml` and `settings.py`.

## Target Audience
- Data science researchers and F1 strategy enthusiasts.
