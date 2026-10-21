# F1 Strategy Prediction

This project aims to build a machine learning model to predict the next pit stop lap (`PitNextLap`) for Formula 1 cars. By analyzing historical race data, we aim to identify patterns in driver behavior, car degradation, and tactical decisions to provide reliable predictions for race simulations.

## Problem Statement
Predicting when a driver will pit is crucial for understanding race dynamics. Simple rule-based approaches often fail to capture the complex interplay between tire wear, fuel load, and tactical positioning. Our solution uses machine learning to incorporate:
- **Historical Performance:** Previous laps' positions and pace changes (`LapTime_Delta`).
- **Degradation Metrics:** Real-time tracking of car performance drops (TyreLife, Normalized_TyreLife).
- **Tire Compounds:** Analysis of the specific tire types fitted (including INTERMEDIATE and WET).
- **Rolling Averages:** Smoothing out noise in lap-by-lap data for more stable features.

## Features
- **Advanced Preprocessing:** Implementation of lagged features and rolling averages grouped by Year, Race, and Driver.
- **Hyperparameter Optimization:** Automated tuning using Optuna to find optimal model parameters.
- **Experiment Tracking:** Full integration with MLflow to log parameters, metrics, and model artifacts for reproducibility.
- **Time-Series Aware Splitting:** Ensures no data leakage by using chronological splits (Train: 2022-2023, Val: 2024, Test: 2025).

## Tech Stack
- **Language:** Python 3.12
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM, CatBoost
- **Optimization:** Optuna
- **Experiment Tracking:** MLflow
- **Package Management:** `uv`

## Project Structure
The project follows a modular design to ensure maintainability and reusability:
- `src/f1_strategy_dataset/__init__.py`: Package initialization.
- `src/f1_strategy_dataset/settings.py`: Centralized configuration loading and MLflow tracking setup.
- `src/f1_strategy_dataset/config.yaml`: Project configuration (data paths, hyperparameter defaults, Optuna settings, MLflow tracking URI).
- `src/f1_strategy_dataset/preprocessing.py`: Data cleaning, feature engineering (lags, rolling windows), and encoding.
- `src/f1_strategy_dataset/hyperparameter_search_optuna.py`: Manages the Optuna study and objective functions.
- `src/f1_strategy_dataset/hyperparameter_search.py`: Alternative `RandomizedSearchCV`-based tuning.
- `src/f1_strategy_dataset/mlflow_utils.py`: Centralizes MLflow logging logic.
- `src/f1_strategy_dataset/main.py`: Orchestrates the entire pipeline from preprocessing to training and evaluation.
- `src/f1_strategy_dataset/load_and_predict.py`: Standalone inference script to load the best model and generate predictions.

## Getting Started

### Prerequisites
- Python 3.12+
- `uv` installed on your system

### Installation
1. Clone the repository.
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

### Running the Pipeline
The entry point `f1-strategy-dataset` accepts a `--mode` argument with three options:

| Mode | Description |
|------|-------------|
| `tune` | Run the Optuna hyperparameter search and save the best parameters to `data/best_params.json`. |
| `train` | Load the saved best parameters, train the final model, and evaluate it on the validation set. |
| `evaluate` | Load the best model (highest `val_auc_roc`) from MLflow and evaluate it on the test set. |

To run the full training and optimization pipeline:
```bash
uv run f1-strategy-dataset --mode tune
uv run f1-strategy-dataset --mode train
uv run f1-strategy-dataset --mode evaluate
```

### Inference
To load the best model and generate predictions on the 2025 test set (or a custom CSV):
```bash
uv run python src/f1_strategy_dataset/load_and_predict.py
uv run python src/f1_strategy_dataset/load_and_predict.py --new-data custom.csv --output outputs/predictions.csv
```

## Methodology
- **Feature Engineering:** We use `.shift()` and `.rolling()` operations to create temporal features. Notably, we use `LapTime_Delta` instead of raw `LapTime (s)` to capture pace change and avoid high collinearity.
- **Encoding:** Tire compounds are converted to categorical types and processed using `OneHotEncoder`. All five compound types (SOFT, MEDIUM, HARD, INTERMEDIATE, WET) are supported.
- **Model Selection:** LightGBM is our primary model due to its efficiency and superior handling of tabular data and categorical features.
- **Evaluation:** Models are evaluated based on Precision and AUC-ROC.

## Progress
- [x] Project structure and memory bank initialized.
- [x] Preprocessing pipeline implemented (lagged features, rolling averages).
- [x] Optuna hyperparameter search integrated.
- [x] MLflow logging utilities functional.
- [x] Evaluate mode added to load and assess the best MLflow model.
- [x] Standalone inference script (`load_and_predict.py`) implemented.
- [x] Console script entry point (`f1-strategy-dataset`) wired in `pyproject.toml`.
- [x] Baseline evaluate run captured: Test Precision 0.8205, Test AUC-ROC 0.7781.
- [x] Feature ingestion fixed: TyreLife, Normalized_TyreLife, Stint, Position_Change, RaceProgress now included; lagged/rolling TyreLife features added.
- [x] Compound encoding fixed: INTERMEDIATE and WET compounds no longer coerced to NaN; all 5 compound types encoded, NaN compounds filled with most common.
- [x] `strategy.ipynb` EDA extended with interaction analysis (Compound × Degradation, Position × LapNumber) and a LogisticRegression vs LightGBM vs CatBoost baseline. Reproduced test-set AUC: LightGBM ~0.952, CatBoost ~0.932, LogisticRegression ~0.818 (tree margin ~0.13 AUC), empirically justifying tree models.
- [ ] Optimization of the Optuna search space for better AUC-ROC.
- [ ] Final evaluation on the 2025 test set.
