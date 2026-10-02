# Active Context

## Current Focus
- Initializing the project structure and memory bank.
- Reviewing existing preprocessing and hyperparameter search logic.

## Recent Changes
- Created memory bank files.
- Analyzed `main.py`, `preprocessing.py`, and `hyperparameter_search_optuna.py`.

## Next Steps
- Verify the data quality in `f1_strategy_dataset_v4.csv`.
- Experiment with additional lagged features.
- Refine the Optuna search space.

## Important Decisions
- Using LightGBM as the primary model due to its efficiency with tabular data.
- Implementing a time-series aware split (2022-2023 train, 2024 val, 2025 test).
