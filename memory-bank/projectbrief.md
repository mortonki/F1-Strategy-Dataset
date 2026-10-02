# Project Brief: F1 Strategy Prediction

## Goals
- Build a machine learning model to predict the next pit stop lap (`PitNextLap`) for F1 cars.
- Use historical race data to identify patterns in driver behavior and car degradation.
- Implement a robust hyperparameter tuning pipeline using Optuna.
- Track all experiments using MLflow for reproducibility and comparison.

## Scope
- Data preprocessing including lagged features and rolling averages.
- Model training using LightGBM and CatBoost.
- Evaluation metrics: Precision and AUC-ROC.
- Time-series aware data splitting.

## Target Audience
- Data science researchers and F1 strategy enthusiasts.
