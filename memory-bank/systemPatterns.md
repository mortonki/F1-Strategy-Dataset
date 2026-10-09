# System Patterns

## Architecture
- **Modular Design:** Logic is separated into distinct modules:
  - `__init__.py`: Package initialization.
  - `settings.py`: Centralized configuration loading (`load_config`) and MLflow tracking setup (`init_mlflow`).
  - `config.yaml`: Project configuration (data paths, LightGBM defaults, Optuna settings, MLflow tracking URI).
  - `preprocessing.py`: Handles data cleaning, feature engineering (lags, rolling windows), and encoding.
  - `hyperparameter_search_optuna.py`: Manages the Optuna study and objective functions.
  - `hyperparameter_search.py`: Alternative `RandomizedSearchCV`-based tuning.
  - `mlflow_utils.py`: Centralizes MLflow logging logic.
  - `main.py`: Orchestrates the entire pipeline (`tune` / `train` / `evaluate` modes).
  - `load_and_predict.py`: Standalone inference script mirroring `evaluate()`.

## Key Technical Decisions
- **Feature Engineering:** Uses `.shift()` and `.rolling()` grouped by `Year`, `Race`, and `Driver` to ensure temporal consistency.
- **Encoding:** `OrdinalEncoder` is used for tire compounds after converting them to categorical types.
- **Model Selection:** LightGBM is chosen for its speed and handling of categorical features.
- **Experiment Tracking:** Every run logs parameters, metrics, and the final model artifact to MLflow.
- **Config-Driven:** `config.yaml` + `settings.py` centralize paths and the MLflow tracking URI; relative URIs are resolved against the repo root.

## Component Relationships
- `main.py` calls `preprocess_f1_data` from `preprocessing.py`.
- `main.py` calls `run_optuna_search` from `hyperparameter_search_optuna.py`.
- Both search and main training call utilities from `mlflow_utils.py`.
- `load_and_predict.py` imports `load_data` from `main.py`, `load_best_model` from `mlflow_utils.py`, and `preprocess_f1_data` from `preprocessing.py`.
- `main.py`'s `evaluate()` calls `load_best_model(config)` to restore the best model from MLflow.
