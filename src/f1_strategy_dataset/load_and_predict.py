#!/usr/bin/env python
"""Utility script to load a trained F1 strategy model from MLflow and make predictions on new data.

The script performs the following steps:

1. **Determine the MLflow run** – If a run ID is supplied via the ``--run-id`` flag the
   script uses that run.  Otherwise it queries the experiment for the most recent run.
2. **Load the model** – The model is loaded from the ``best_model`` artifact that
   is logged in :pyfunc:`main.main`.
3. **Re‑create the preprocessing pipeline** – The training data is re‑loaded and
   split in the same way as in :pyfunc:`main.main`.  ``preprocess_f1_data`` is
   called with ``is_training=True`` to obtain a fitted ``SimpleImputer`` and
   ``OrdinalEncoder``.
4. **Preprocess the new data** – The new data file is read, the same preprocessing
   steps are applied using the fitted transformer objects, and the feature matrix
   is produced.
5. **Predict** – The model predicts the target and the predictions are written
   to ``predictions.csv`` (or a user‑supplied output path).

The script is intentionally lightweight and does not depend on any external
configuration beyond the local MLflow tracking URI (``sqlite:///mlflow.db``) and the
``f1_strategy_dataset_v4.csv`` training file.
"""

from __future__ import annotations
from f1_strategy_dataset.main import load_data



def main() -> None:
    parser = argparse.ArgumentParser(description="Load an MLflow model and predict on new F1 data.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID containing the best model. If omitted, the latest run is used.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="f1_strategy_dataset_v4.csv",
        help="Path to the full training dataset used for fitting the preprocessors.",
    )
    parser.add_argument(
        "--new-data",
        type=str,
        required=True,
        help="Path to the CSV file containing new observations to predict.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="File to write predictions to.",
    )
    args = parser.parse_args()

    # Determine run ID
    run_id = args.run_id or _get_latest_run_id()
    print(f"Using MLflow run: {run_id}")

    # Load model
    model = load_model(run_id)
    print("Model loaded.")

    # Load training data to fit preprocessors
    train_df = pd.read_csv(args.data_path)
    # Split train/val/test as in main.py to get the same training split
    train_df = train_df[train_df["Year"].isin([2022, 2023])].copy()
    imputer, encoder = prepare_preprocessors(train_df)
    print("Preprocessors fitted.")

    # Load new data and preprocess
    new_df = pd.read_csv(args.new_data)
    X_new = preprocess_new_data(new_df, imputer, encoder)
    print("New data preprocessed.")

    # Predict
    predictions = model.predict(X_new)
    print("Prediction completed.")

    # Save predictions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
