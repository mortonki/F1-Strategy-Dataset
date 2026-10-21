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
- [x] Baseline evaluate run captured: Test Precision 0.8205, Test AUC-ROC 0.7781.
- [x] Feature ingestion fixed: TyreLife, Normalized_TyreLife, Stint, Position_Change, RaceProgress now included; lagged/rolling TyreLife features added.
- [x] Compound encoding fixed: INTERMEDIATE and WET compounds no longer coerced to NaN; all 5 compound types encoded, NaN compounds filled with most common.
- [x] `strategy.ipynb` EDA extended with interaction analysis (Compound × Degradation, Position × LapNumber) and a LogisticRegression vs LightGBM vs CatBoost baseline. Reproduced test-set AUC: LightGBM ~0.952, CatBoost ~0.932, LogisticRegression ~0.818 (tree margin ~0.13 AUC), empirically justifying tree models.

## What's Left
- [ ] Detailed exploration of `f1_strategy_dataset_v4.csv` (e.g., INTERMEDIATE/WET compounds not covered by the SOFT/MEDIUM/HARD encoding).
- [ ] Optimization of the Optuna search space for better AUC-ROC.
- [ ] Final evaluation on the 2025 test set.

## Known Issues
- The 2023 pit-rate distribution shift (~3.1% vs ~32-33% in 2022/2024) causes train/val/test target imbalance. Mitigated via class weighting in the model.
- Two MLflow models are persisted with identical `val_auc_roc` (0.7517); the "best" model selection is therefore arbitrary between them.

## Evolution of Decisions
- Switched to LightGBM as primary model for better performance on tabular data.
- Decided to use a time-series aware split to prevent data leakage from future years.
- Centralized configuration into `config.yaml` + `settings.py` to remove hardcoded MLflow tracking setup.
- Added TyreLife (corr ~0.266, strongest predictor) and lagged/rolling tyre-life features to recover predictive signal.
- Fixed compound encoding to preserve INTERMEDIATE/WET signal (was being dropped).
