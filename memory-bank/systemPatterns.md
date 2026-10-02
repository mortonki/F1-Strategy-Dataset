# System Patterns

## Architecture
- **Modular Design:** Logic is separated into distinct modules:
  - `preprocessing.py`: Handles data cleaning, feature engineering (lags, rolling windows), and encoding.
  - `hyperparameter_search_optuna.py`: Manages the Optuna study and objective functions.
  - `mlflow_utils.py`: Centralizes MLflow logging logic.
  - `main.py`: Orchestrates the entire pipeline.

## Key Technical Decisions
- **Feature Engineering:** Uses `.shift()` and `.rolling()` grouped by `Year`, `Race`, and `Driver` to ensure temporal consistency.
- **Encoding:** `OrdinalEncoder` is used for tire compounds after converting them to categorical types.
- **Model Selection:** LightGBM is chosen for its speed and handling of categorical features.
- **Experiment Tracking:** Every run logs parameters, metrics, and the final model artifact to MLflow.

## Component Relationships
- `main.py` calls `preprocess_f1_data` from `preprocessing.py`.
- `main.py` calls `run_optuna_search` from `hyperparameter_search_optuna.py`.
- Both search and main training call utilities from `mlflow_utils.py`.
