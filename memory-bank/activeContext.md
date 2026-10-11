# Active Context

## Current Focus
- The pipeline is complete end-to-end: `tune` (Optuna) → `train` (fit with saved params) → `evaluate` (load best model from MLflow, assess on test set).
- `load_and_predict.py` is a standalone inference script that mirrors `evaluate()` and writes predictions to CSV.
- Two LightGBM models are persisted in MLflow (`mlruns/mlflow.db`): `m-4fb15c395ddd4e00a8f962d0519a8b8f` and `m-7115f37d76e1450190ea69373d94001f`, both with `val_auc_roc=0.7517`.
- Baseline evaluate run: Test Precision 0.8205, Test AUC-ROC 0.7781.
- Feature ingestion fixed: TyreLife, Normalized_TyreLife, Stint, Position_Change, RaceProgress now included; lagged/rolling TyreLife features added.
- Compound encoding fixed: INTERMEDIATE and WET compounds no longer coerced to NaN; all 5 compound types encoded, NaN compounds filled with most common.

## Recent Changes
- Reworked `load_and_predict.py` to mirror `evaluate()`: loads the best `val_auc_roc` model via `load_best_model`, reconstructs preprocessing (fit on train, transform test), predicts, and writes a `prediction` CSV. Added `--new-data` and `--output` args.
- Centralized config via `settings.py` (`load_config`, `get_tracking_uri`, `init_mlflow`) and moved MLflow tracking setup out of `main.py`/`mlflow_utils.py`.
- Console script entry point `f1-strategy-dataset` wired to `f1_strategy_dataset.main:main` in `pyproject.toml`.
- Added TyreLife, Normalized_TyreLife, Stint, Position_Change, RaceProgress features and lagged/rolling TyreLife features to `preprocessing.py`.
- Fixed compound encoding in `preprocessing.py` to include all 5 compound types (SOFT/MEDIUM/HARD/INTERMEDIATE/WET) and fill NaN compounds with the most common.

## Next Steps
- Optimization of the Optuna search space for better AUC-ROC.
- Re-run `tune` → `train` → `evaluate` to confirm pit predictions recover with the new features.
- Final evaluation on the 2025 test set.

## Important Decisions
- Using LightGBM as the primary model due to its efficiency with tabular data.
- Implementing a time-series aware split (2022-2023 train, 2024 val, 2025 test).
- Preprocessing contract: `preprocess_f1_data(df, imputer, encoder, is_training=True)` fits and returns `(X, y, imputer, encoder)`; `is_training=False` transforms and returns `(X, y)`.
- Config-driven: `config.yaml` holds data paths, LightGBM defaults, Optuna settings, and the MLflow tracking URI; `settings.py` resolves relative URIs against the repo root.
- Package name is `f1_strategy_dataset` (underscores) on disk, while the `pyproject.toml` distribution name and console script are `f1-strategy-dataset` (hyphens).
- 2023 pit-rate distribution shift (~3.1% vs ~32-33% in 2022/2024) mitigated via class weighting in the model.
