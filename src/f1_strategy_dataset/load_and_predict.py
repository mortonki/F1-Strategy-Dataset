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
from typing import Any


import argparse
import os
from pathlib import Path

import mlflow
from mlflow import sklearn
import pandas as pd
from f1_strategy_dataset.mlflow_utils import TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from f1_strategy_dataset.preprocessing import preprocess_f1_data


def _get_latest_run_id(experiment_name: str = "F1 Strategy Prediction") -> str:
    """Return the run ID of the most recent run in the given experiment.

    MLflow's :pyfunc:`mlflow.search_runs` API expects an ``experiment_ids``
    argument (a list of integer IDs).  The original implementation passed a
    list of experiment names which caused a ``TypeError``.  This helper now
    resolves the experiment name to its numeric ID using
    :pyfunc:`mlflow.get_experiment_by_name` and then queries for runs.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if len(runs) == 0:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")
    
    # Ensure runs is a DataFrame to support sort_values and satisfy type checkers
    df_runs = pd.DataFrame(runs)
    latest_run = df_runs.sort_values("start_time", ascending=False).iloc[0]
    return latest_run["run_id"]


def load_model(run_id: str) -> Any:
    """Load the best model from the specified MLflow run.

    The artifact path used during training is ``best_model``.
    """
    return sklearn.load_model(f"runs:/{run_id}/best_model")


def prepare_preprocessors(train_df: pd.DataFrame) -> tuple[SimpleImputer, OrdinalEncoder]:
    """Fit the imputer and encoder on the training data.

    The function mirrors the logic in :pyfunc:`main.main` – it creates a
    ``SimpleImputer`` with ``most_frequent`` strategy and an ``OrdinalEncoder``
    for the ``Compound`` column.  The fitted objects are returned.
    """
    imputer = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder()
    # ``preprocess_f1_data`` returns X, y, imputer, encoder when is_training=True
    _, _, fitted_imputer, fitted_encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    return fitted_imputer, fitted_encoder


def preprocess_new_data(df: pd.DataFrame, imputer: SimpleImputer, encoder: OrdinalEncoder) -> pd.DataFrame:
    """Apply the same preprocessing steps to new data.

    The function expects the new data to contain the same columns as the
    training data.  If the target column ``PitNextLap`` is missing it is
    ignored.
    """
    # Ensure the target column is present for consistency with the training
    # pipeline.  If it is missing we simply drop it.
    if "PitNextLap" in df.columns:
        X, _ = preprocess_f1_data(df, imputer, encoder, is_training=False)
    else:
        # Drop the target if present, then run preprocessing
        df_copy = df.copy()
        df_copy = df_copy.drop(columns=[col for col in ["PitNextLap"] if col in df_copy.columns])
        X, _ = preprocess_f1_data(df_copy, imputer, encoder, is_training=False)
    return X


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
