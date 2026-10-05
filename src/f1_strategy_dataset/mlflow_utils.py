"""
MLflow Experiment Tracking Utilities for F1 Strategy Prediction

This module provides utilities for tracking experiments using MLflow,
including parameter logging, metric logging, and model logging.
"""

import mlflow
from mlflow import sklearn
import pandas as pd
import numpy as np
import logging
import os

# Suppress MLflow dependency export log
logging.getLogger('mlflow.utils.uv_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)

# Global tracking URI - can be overridden by MLFLOW_TRACKING_URI environment variable
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)


def init_experiment(experiment_name: str = "F1 Strategy Prediction") -> dict:
    """
    Initialize MLflow experiment tracking.
    
    Args:
        experiment_name: Name of the experiment to track
        
    Returns:
        Experiment info dictionary
    """
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    # Get artifact URI using the experiment ID
    artifact_uri = mlflow.get_artifact_uri(experiment_id)
    return {
        "name": experiment.name,
        "experiment_id": experiment_id,
        "creation_time": experiment.creation_time,
        "lifecycle_stage": experiment.lifecycle_stage,
        "artifact_uri": artifact_uri
    }


def log_parameters(params: dict) -> None:
    """
    Log hyperparameters as MLflow parameters.
    
    Args:
        params: Dictionary of hyperparameters to log
    """
    for key, value in params.items():
        if value is not None:
            mlflow.log_param(key, str(value))


def log_metrics(metrics_dict: dict) -> None:
    """
    Log metrics as MLflow metrics.
    
    Args:
        metrics_dict: Dictionary of metric names to values
    """
    mlflow.log_metrics(metrics_dict)


def log_experiment_info(experiment_info: dict) -> None:
    """
    Log experiment metadata as MLflow parameters.
    
    Args:
        experiment_info: Dictionary containing experiment information
    """
    mlflow.log_params({
        "experiment_name": experiment_info.get("name", "Unknown"),
        "experiment_id": experiment_info.get("experiment_id", "Unknown"),
        "creation_time": str(experiment_info.get("creation_time", 0)),
        "lifecycle_stage": experiment_info.get("lifecycle_stage", "Unknown"),
        "artifact_uri": experiment_info.get("artifact_uri", "Unknown")
    })


def log_data_stats(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> None:
    """
    Log dataset statistics as MLflow parameters and metrics.
    
    Args:
        df: Original DataFrame with data statistics
        X: Feature matrix
        y: Target vector
    """
    # Dataset info
    mlflow.log_params({
        "total_rows": len(df),
        "feature_count": X.shape[1],
        "target_count": len(y)
    })
    
    # Year range
    if "Year" in df.columns:
        mlflow.log_params({
            "year_min": df["Year"].min(),
            "year_max": df["Year"].max()
        })
    
    # Class distribution
    unique_classes = np.unique(y)
    class_counts = {str(c): int(np.sum(y == c)) for c in unique_classes}
    mlflow.log_params(class_counts)
    
    # Feature info
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    mlflow.log_param("feature_names", str(feature_names))


def log_model_info(model, X_train, y_train) -> None:
    """
    Log model configuration and training info as MLflow parameters.
    
    Args:
        model: Trained sklearn-compatible model
        X_train: Training feature matrix
        y_train: Training target vector
    """
    mlflow.log_params({
        "model_type": type(model).__name__,
        "n_features": X_train.shape[1],
        "n_samples": len(y_train)
    })
    
    # Model-specific parameters (for LightGBM)
    params_to_log = {}
    for attr in ["n_estimators", "learning_rate", "num_leaves", "random_state", "n_jobs"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            params_to_log[attr] = str(val) if val is not None else None
            
    mlflow.log_params({k: v for k, v in params_to_log.items() if v is not None})


def log_model(model, name: str = "model") -> None:
    """
    Log the trained model as an MLflow artifact.
    
    Args:
        model: Trained sklearn-compatible model
        name: Name within the run to store the model
    """
    sklearn.log_model(model, name=name)


def log_all(model, params: dict, metrics_dict: dict) -> None:
    """
    Log parameters, metrics, and model in a single operation.
    
    Args:
        model: Trained sklearn-compatible model
        params: Dictionary of hyperparameters
        metrics_dict: Dictionary of metric names to values
    """
    log_parameters(params)
    log_metrics(metrics_dict)
    log_model(model, name="model")
