# Tech Context

## Technologies Used
- **Language:** Python 3.12
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM, CatBoost
- **Optimization:** Optuna
- **Experiment Tracking:** MLflow
- **Package Management:** `uv`

## Development Setup
- Environment managed by `uv`.
- Running commands via `uv run`.
- Standardized random seed: `42`.
- Package name is `f1-strategy-dataset` (hyphens) in `pyproject.toml`; the importable package directory is `f1_strategy_dataset` (underscores).
- Console script entry point `f1-strategy-dataset` maps to `f1_strategy_dataset.main:main`.
- Configuration is centralized in `config.yaml` and loaded via `settings.load_config`; MLflow tracking is initialized via `settings.init_mlflow`.

## Constraints
- Must maintain strict separation between training, validation, and test sets.
- No fitting transformers on validation or test sets.
- Ensure reproducibility by fixing random states.
- Config-driven: relative MLflow tracking URIs are resolved against the repo root by `settings.py`.
