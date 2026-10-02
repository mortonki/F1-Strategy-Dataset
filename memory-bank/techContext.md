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

## Constraints
- Must maintain strict separation between training, validation, and test sets.
- No fitting transformers on validation or test sets.
- Ensure reproducibility by fixing random states.
