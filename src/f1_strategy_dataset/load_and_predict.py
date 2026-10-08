#!/usr/bin/env python
"""Utility script to load a trained F1 strategy model from MLflow and make predictions.

Mirrors the ``evaluate()`` function in ``main.py``:
1. Load the best performing model (highest val_auc_roc) from MLflow.
2. Load the dataset and split into train/test using ``load_data``.
3. Fit a SimpleImputer and OrdinalEncoder on the training data via ``preprocess_f1_data``.
4. Preprocess the test data using the fitted transformers.
5. Predict and write the predictions to a CSV file.

An optional ``--new-data`` argument can be provided to predict on a custom CSV file
instead of the test set.
"""
from __future__ import annotations

import argparse
import mlflow
from mlflow import sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from f1_strategy_dataset.main import load_data
from f1_strategy_dataset.mlflow_utils import load_best_model
from f1_strategy_dataset.preprocessing import preprocess_f1_data
from f1_strategy_dataset import settings
from f1_strategy_dataset.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the best MLflow model and predict on F1 data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--new-data",
        type=str,
        default=None,
        help="Path to a CSV file containing new observations to predict. "
        "If omitted, the test set (year 2025) from load_data is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.csv",
        help="File to write predictions to (defaults to outputs/predictions.csv).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    settings.init_mlflow(config)

    # Load the best performing model (highest val_auc_roc) from MLflow.
    model = load_best_model(config)
    print(f"Loaded best model: {type(model).__name__}")

    # Load the dataset and split into train/test using load_data.
    _, train_df, _, test_df = load_data(config["DATA_PATH"])

    # Reconstruct preprocessing objects (fit on train, transform test).
    imputer = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder()
    preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    X_test, y_test = preprocess_f1_data(test_df, imputer, encoder, is_training=False)

    # Predict.
    y_pred = np.array(model.predict(X_test))
    print("Prediction completed.")

    # Save predictions.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": y_pred}).to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
