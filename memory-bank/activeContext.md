# Active Context

## Current Focus
- Implementing `evaluate`-mode model loading that loads the best `val_auc_roc` model and fails fast if its artifacts are missing on disk.

## Recent Changes
- Created memory bank files.
- Analyzed `main.py`, `preprocessing.py`, and `hyperparameter_search_optuna.py`.
- Added `load_best_model(config)` to `mlflow_utils.py`: queries the MLflow DB for the highest `val_auc_roc` model, loads it via `mlflow.sklearn.load_model`, raises `MlflowException` if artifacts are missing on disk / `ValueError` if no metric.
- Added `evaluate(config)` to `main.py`: loads the best model, reconstructs preprocessing (fit on train, transform test), predicts on the test set, logs test precision/AUC-ROC.
- Wired `main()`'s `evaluate` branch to call `evaluate(config)` and return exit code 1 on failure (stopped the fall-through to `train_and_evaluate`). Changed `if __name__ == "__main__": main()` to `sys.exit(main())`.

## Next Steps
- Verify the data quality in `f1_strategy_dataset_v4.csv`.
- Experiment with additional lagged features.
- Refine the Optuna search space.

## Important Decisions
- Using LightGBM as the primary model due to its efficiency with tabular data.
- Implementing a time-series aware split (2022-2023 train, 2024 val, 2025 test).
- The best model by `val_auc_roc` (`m-ddc97f979661402c8fb7af982700f712`, auc=0.7892) was originally missing on disk; the evaluate mode deliberately surfaces this as a failure.
- Preprocessing contract: `preprocess_f1_data(df, imputer, encoder, is_training=True)` fits; `is_training=False` transforms.
