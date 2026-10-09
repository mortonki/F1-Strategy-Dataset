# Active Context

## Current Focus
- The pipeline is complete end-to-end: `tune` (Optuna) → `train` (fit with saved params) → `evaluate` (load best model from MLflow, assess on test set).
- `load_and_predict.py` is a standalone inference script that mirrors `evaluate()` and writes predictions to CSV.
- Two LightGBM models are persisted in MLflow (`mlruns/mlflow.db`): `m-4fb15c395ddd4e00a8f962d0519a8b8f` and `m-7115f37d76e1450190ea69373d94001f`, both with `val_auc_roc=0.7517`.
- Open data-quality item: the preprocessing only encodes SOFT/MEDIUM/HARD compounds, but the data also contains INTERMEDIATE and WET (dropped and imputed to 0).

## Recent Changes
- Reworked `load_and_predict.py` to mirror `evaluate()`: loads the best `val_auc_roc` model via `load_best_model`, reconstructs preprocessing (fit on train, transform test), predicts, and writes a `prediction` CSV. Added `--new-data` and `--output` args.
- Centralized config via `settings.py` (`load_config`, `get_tracking_uri`, `init_mlflow`) and moved MLflow tracking setup out of `main.py`/`mlflow_utils.py`.
- Console script entry point `f1-strategy-dataset` wired to `f1_strategy_dataset.main:main` in `pyproject.toml`.

## Next Steps
- Investigate the INTERMEDIATE/WET compound handling (currently coerced to NaN then imputed to 0).
- Experiment with additional lagged features.
- Refine the Optuna search space for better AUC-ROC.

## Important Decisions
- Using LightGBM as the primary model due to its efficiency with tabular data.
- Implementing a time-series aware split (2022-2023 train, 2024 val, 2025 test).
- Preprocessing contract: `preprocess_f1_data(df, imputer, encoder, is_training=True)` fits and returns `(X, y, imputer, encoder)`; `is_training=False` transforms and returns `(X, y)`.
- Config-driven: `config.yaml` holds data paths, LightGBM defaults, Optuna settings, and the MLflow tracking URI; `settings.py` resolves relative URIs against the repo root.
- Package name is `f1_strategy_dataset` (underscores) on disk, while the `pyproject.toml` distribution name and console script are `f1-strategy-dataset` (hyphens).
