# F1 Strategy Prediction

This project aims to build a machine learning model to predict the next pit stop lap (`PitNextLap`) for Formula 1 cars. By analyzing historical race data, we aim to identify patterns in driver behavior, car degradation, and tactical decisions to provide reliable predictions for race simulations.

## Problem Statement
Predicting when a driver will pit is crucial for understanding race dynamics. Simple rule-based approaches often fail to capture the complex interplay between tire wear, fuel load, and tactical positioning. Our solution uses machine learning to incorporate:
- **Historical Performance:** Previous laps' positions and lap times.
- **Degradation Metrics:** Real-time tracking of car performance drops.
- **Tire Compounds:** Analysis of the specific tire types fitted.
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
- `src/f1-strategy-dataset/preprocessing.py`: Data cleaning, feature engineering (lags, rolling windows), and encoding.
- `src/f1-strategy-dataset/hyperparameter_search_optuna.py`: Manages the Optuna study and objective functions.
- `src/f1-strategy-dataset/mlflow_utils.py`: Centralizes MLflow logging logic.
- `src/f1-strategy-dataset/main.py`: Orchestrates the entire pipeline from preprocessing to training and evaluation.

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
To run the full training and optimization pipeline:
```bash
uv run f1-strategy-dataset
```

## Methodology
- **Feature Engineering:** We use `.shift()` and `.rolling()` operations to create temporal features while maintaining consistency across drivers and races.
- **Encoding:** Tire compounds are converted to categorical types and processed using `OrdinalEncoder`.
- **Model Selection:** LightGBM is our primary model due to its efficiency and superior handling of tabular data and categorical features.
- **Evaluation:** Models are evaluated based on Precision and AUC-ROC.

## Progress
- [x] Project structure and memory bank initialized.
- [x] Basic preprocessing pipeline implemented.
- [x] Optuna hyperparameter search integrated.
- [x] MLflow logging utilities functional.
- [ ] Detailed exploration of `f1_strategy_dataset_v4.csv`.
- [ ] Optimization of Optuna search space for better AUC-ROC.
- [ ] Testing of the inference script `load_and_predict.py`.
- [ ] Final evaluation on the 2025 test set.

