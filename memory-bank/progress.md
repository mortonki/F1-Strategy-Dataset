# Progress

## What Works
- [x] Project structure initialized.
- [x] Memory bank established.
- [x] Preprocessing pipeline (lagged features, rolling averages) implemented.
- [x] Optuna hyperparameter search integrated.
- [x] MLflow logging utilities functional.
- [x] Evaluate mode added to load and assess the best MLflow model.
- [x] Standalone inference script (`load_and_predict.py`) implemented.
- [x] Console script entry point (`f1-strategy-dataset`) wired in `pyproject.toml`.

## What's Left
- [ ] Detailed exploration of `f1_strategy_dataset_v4.csv` to identify further features (e.g., INTERMEDIATE/WET compounds not covered by the SOFT/MEDIUM/HARD encoding).
- [ ] Optimization of the Optuna search space for better AUC-ROC.
- [ ] Final evaluation on the 2025 test set.

## Known Issues
- The preprocessing only encodes SOFT/MEDIUM/HARD compounds; INTERMEDIATE and WET are coerced to NaN and imputed to 0, so they carry no signal.
- Two MLflow models are persisted with identical `val_auc_roc` (0.7517); the "best" model selection is therefore arbitrary between them.

## Evolution of Decisions
- Switched to LightGBM as primary model for better performance on tabular data.
- Decided to use a time-series aware split to prevent data leakage from future years.
- Centralized configuration into `config.yaml` + `settings.py` to remove hardcoded MLflow tracking setup.
