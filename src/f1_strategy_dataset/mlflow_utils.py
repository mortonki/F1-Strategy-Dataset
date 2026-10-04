"""
MLflow Experiment Tracking Utilities for F1 Strategy Prediction

This module provides utilities for tracking experiments using MLflow,
including parameter logging, metric logging, and model logging.
"""

import mlflow
from mlflow.sklearn import log_model
import argparse
import pandas as pd
import numpy as np
import logging

# Suppress MLflow dependency export log
logging.getLogger('mlflow.utils.uv_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)


# Global tracking URI for local file system
TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)


def init_experiment(experiment_name: str = "F1 Strategy Prediction") -> dict:
    """
    Initialize MLflow experiment tracking.
    
    Args:
        experiment_name: Name of the experiment to track
        
    Returns:
        Experiment info dictionary
    """
    mlflow.set_tracking_uri(TRACKING_URI)
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


def log_parameters(args: argparse.Namespace, run_id: str | None = None) -> None:
    """
    Log hyperparameters as MLflow parameters.
    
    Args:
        args: Namespace object containing hyperparameters
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("learning_rate", args.learning_rate)
            mlflow.log_param("num_leaves", args.num_leaves)
            mlflow.log_param("random_state", args.random_state)
            if args.n_jobs is not None:
                mlflow.log_param("n_jobs", str(args.n_jobs))
    else:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_leaves", args.num_leaves)
        mlflow.log_param("random_state", args.random_state)
        if args.n_jobs is not None:
            mlflow.log_param("n_jobs", str(args.n_jobs))


def log_metrics(metrics_dict: dict, run_id: str | None = None) -> None:
    """
    Log metrics as MLflow metrics.
    
    Args:
        metrics_dict: Dictionary of metric names to values
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, metric_value)
    else:
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value)


def log_experiment_info(experiment_info: dict, run_id: str | None = None) -> None:
    """
    Log experiment metadata as MLflow parameters.
    
    Args:
        experiment_info: Dictionary containing experiment information
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            mlflow.log_param("experiment_name", experiment_info.get("name", "Unknown"))
            mlflow.log_param("experiment_id", experiment_info.get("experiment_id", "Unknown"))
            mlflow.log_param("creation_time", str(experiment_info.get("creation_time", 0)))
            mlflow.log_param("lifecycle_stage", experiment_info.get("lifecycle_stage", "Unknown"))
            mlflow.log_param("artifact_uri", experiment_info.get("artifact_uri", "Unknown"))
    else:
        mlflow.log_param("experiment_name", experiment_info.get("name", "Unknown"))
        mlflow.log_param("experiment_id", experiment_info.get("experiment_id", "Unknown"))
        mlflow.log_param("creation_time", str(experiment_info.get("creation_time", 0)))
        mlflow.log_param("lifecycle_stage", experiment_info.get("lifecycle_stage", "Unknown"))
        mlflow.log_param("artifact_uri", experiment_info.get("artifact_uri", "Unknown"))


def log_data_stats(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, run_id: str | None = None) -> None:
    """
    Log dataset statistics as MLflow parameters and metrics.
    
    Args:
        df: Original DataFrame with data statistics
        X: Feature matrix
        y: Target vector
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            # Dataset info
            mlflow.log_param("total_rows", len(df))
            mlflow.log_param("feature_count", X.shape[1])
            mlflow.log_param("target_count", len(y))
            
            # Year range
            if "Year" in df.columns:
                mlflow.log_param("year_min", df["Year"].min())
                mlflow.log_param("year_max", df["Year"].max())
            
            # Class distribution
            unique_classes = np.unique(y)
            class_counts = {str(c): int(np.sum(y == c)) for c in unique_classes}
            mlflow.log_params(class_counts)
            
            # Feature info
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            mlflow.log_params({"feature_names": feature_names})
    else:
        # Dataset info
        mlflow.log_param("total_rows", len(df))
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_param("target_count", len(y))
        
        # Year range
        if "Year" in df.columns:
            mlflow.log_param("year_min", df["Year"].min())
            mlflow.log_param("year_max", df["Year"].max())
        
        # Class distribution
        unique_classes = np.unique(y)
        class_counts = {str(c): int(np.sum(y == c)) for c in unique_classes}
        mlflow.log_params(class_counts)
        
        # Feature info
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        mlflow.log_params({"feature_names": feature_names})


def log_model_info(model, X_train, y_train, run_id: str | None = None) -> None:
    """
    Log model configuration and training info as MLflow parameters.
    
    Args:
        model: Trained sklearn-compatible model
        X_train: Training feature matrix
        y_train: Training target vector
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            # Model info
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", len(y_train))
            
            # Model-specific parameters (for LightGBM)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "learning_rate"):
                mlflow.log_param("learning_rate", model.learning_rate)
            if hasattr(model, "num_leaves"):
                mlflow.log_param("num_leaves", model.num_leaves)
            if hasattr(model, "random_state"):
                mlflow.log_param("random_state", model.random_state)
            if hasattr(model, "n_jobs") and model.n_jobs is not None:
                mlflow.log_param("n_jobs", str(model.n_jobs))
    else:
        # Model info
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", len(y_train))
        
        # Model-specific parameters (for LightGBM)
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "learning_rate"):
            mlflow.log_param("learning_rate", model.learning_rate)
        if hasattr(model, "num_leaves"):
            mlflow.log_param("num_leaves", model.num_leaves)
        if hasattr(model, "random_state"):
            mlflow.log_param("random_state", model.random_state)
        if hasattr(model, "n_jobs") and model.n_jobs is not None:
            mlflow.log_param("n_jobs", str(model.n_jobs))


def log_model(model, name: str = "model", run_id: str | None = None) -> None:
    """
    Log the trained model as an MLflow artifact.
    
    Args:
        model: Trained sklearn-compatible model
        name: Name within the run to store the model
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            log_model(model, name=name)
    else:
        log_model(model, name=name)


def log_all(model, args, metrics_dict: dict, run_id: str | None = None) -> None:
    """
    Log parameters, metrics, and model in a single operation.
    
    Args:
        model: Trained sklearn-compatible model
        args: Namespace object containing hyperparameters
        metrics_dict: Dictionary of metric names to values
        run_id: Optional run ID to log to. If None, starts a new run.
    """
    if run_id is None:
        with mlflow.start_run():
            # Log only actual hyperparameters, not all namespace attributes
            hyperparameters = {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "random_state": args.random_state,
                "n_jobs": args.n_jobs
            }
            for key, value in hyperparameters.items():
                if value is not None:
                    mlflow.log_param(key, str(value))
            mlflow.log_metrics(metrics_dict)
            log_model(model, name="model")
    else:
        hyperparameters = {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs
        }
        for key, value in hyperparameters.items():
            if value is not None:
                mlflow.log_param(key, str(value))
        mlflow.log_metrics(metrics_dict)
        log_model(model, name="model")
