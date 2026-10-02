# Progress

## What Works
- [x] Project structure initialized.
- [x] Memory bank established.
- [x] Basic preprocessing pipeline (lagged features, rolling averages) implemented.
- [x] Optuna hyperparameter search integrated.
- [x] MLflow logging utilities functional.

## What's Left
- [ ] Detailed exploration of `f1_strategy_dataset_v4.csv` to identify further features.
- [ ] Optimization of the Optuna search space for better AUC-ROC.
- [ ] Testing of the inference script `load_and_predict.py`.
- [ ] Final evaluation on the 2025 test set.

## Known Issues
- None currently identified.

## Evolution of Decisions
- Switched to LightGBM as primary model for better performance on tabular data.
- Decided to use a time-series aware split to prevent data leakage from future years.
