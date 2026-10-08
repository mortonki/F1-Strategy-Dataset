"""
Hyperparameter Search Module for F1 Strategy Prediction using Optuna

This module provides functions for hyperparameter tuning using Optuna with Bayesian optimization.
Optuna is memory-efficient and state-of-the-art for hyperparameter tuning.
"""

import argparse
import pandas as pd
import numpy as np
import optuna
import mlflow
from mlflow import sklearn
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from f1_strategy_dataset.preprocessing import preprocess_f1_data

# Suppress MLflow dependency export log
logging.getLogger('mlflow.utils.uv_utils').setLevel(logging.ERROR)
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)

trial = optuna.trial.Trial


def create_optuna_study(
    study_name: str = 'f1_strategy_optuna',
    n_trials: int = 50,
    n_jobs: int = -1,
    pruner: str = 'tpg',
    sampler: str = 'tpes'
):
    """
    Create and configure an Optuna study.
    
    Args:
        study_name: Name for the study
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (-1 for all available CPUs)
        pruner: Pruning strategy - 'tpg' (Tree-Structured Pruning) or 'median'
        sampler: Sampler - 'tpes' (Tree-structured Parzen Estimator) or 'hbopt' (Hyperband)
    
    Returns:
        optuna.Study: Configured Optuna study
    """
    study = optuna.create_study(
        study_name=study_name,
        directions=['minimize'],
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=True
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    return study


def objective(
    trial: optuna.trial.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str = 'lgbm',
    n_trials: int = 50,
    n_jobs: int = -1
):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        train_df: Training data
        val_df: Validation data
        model_type: Type of model to use ('lgbm' or 'catboost')
        n_trials: Number of trials
        n_jobs: Number of parallel jobs
    
    Returns:
        float: Negative AUC-ROC (to minimize)
    """
    # Create imputer and encoder
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    
    # Preprocess training data
    X_train, y_train, imputer, encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    
    # Preprocess validation data
    X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=False)

    if model_type == 'lgbm':
        # Get hyperparameters from trial for LightGBM
        n_estimators = trial.suggest_int('n_estimators', low=100, high=1000)
        learning_rate = trial.suggest_float('learning_rate', low=0.01, high=0.2, log=True)
        num_leaves = trial.suggest_int('num_leaves', low=10, high=100)
        max_depth = trial.suggest_int('max_depth', low=3, high=11)
        min_child_samples = trial.suggest_int('min_child_samples', low=10, high=100)
        subsample = trial.suggest_float('subsample', low=0.8, high=1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', low=0.8, high=1.0)
        reg_alpha = trial.suggest_float('reg_alpha', low=0.0, high=1.0)
        reg_lambda = trial.suggest_float('reg_lambda', low=0.0, high=1.0)
        min_data_for_leaf = trial.suggest_int('min_data_for_leaf', low=5, high=20)
        min_data_for_host = trial.suggest_int('min_data_for_host', low=1, high=4)
        feature_fraction = trial.suggest_float('feature_fraction', low=0.7, high=1.0)
        bagging_fraction = trial.suggest_float('bagging_fraction', low=0.7, high=1.0)
        bagging_freq = trial.suggest_int('bagging_freq', low=0, high=3)
        verbose = trial.suggest_int('verbose', low=-1, high=1)
        seed = trial.suggest_int('seed', low=42, high=123)

        # Define LightGBM classifier
        lgb_model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_data_for_leaf=min_data_for_leaf,
            min_data_for_host=min_data_for_host,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=seed
        )
        
        # Train model with early stopping
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],  # Validation data for early stopping
            callbacks=[lgb.early_stopping(stopping_rounds=50)], # Number of early stopping rounds
        )
        model = lgb_model

    elif model_type == 'catboost':
        # Get hyperparameters from trial for CatBoost
        iterations = trial.suggest_int('iterations', low=100, high=1000)
        learning_rate = trial.suggest_float('learning_rate', low=0.01, high=0.2, log=True)
        depth = trial.suggest_int('depth', low=3, high=11)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', low=1e-3, high=10.0, log=True)
        random_seed = trial.suggest_int('random_seed', low=42, high=123)
        loss_function = trial.suggest_categorical('loss_function', ['Logloss', 'AUC'])
        
        # Define CatBoost classifier

        catboost_model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            loss_function=loss_function,
            verbose=0, # Suppress verbose output during tuning
            random_state=random_seed
        )
        
        # Train model
        catboost_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=0
        )
        model = catboost_model
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be 'lgbm' or 'catboost'.")

    # Evaluate on validation set
    y_pred_val = np.asarray(model.predict(X_val))
    probs = np.asarray(model.predict_proba(X_val))
    y_pred_proba_val = probs[:, 1] if hasattr(probs, 'toarray') else probs[:, 1]
    
    val_precision = precision_score(y_val, y_pred_val)
    val_auc_roc = roc_auc_score(y_val, y_pred_proba_val)
    
    # Return negative AUC-ROC (Optuna minimizes)
    return float(-val_auc_roc)


def run_optuna_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str = 'lgbm',
    n_trials: int = 50,
    n_jobs: int = -1,
    pruner: str = 'tpg',
    sampler: str = 'tpes',
    verbose: int = 1
) -> dict:
    """
    Run Optuna-based hyperparameter tuning.

    Args:
        train_df: Training data (pre-split)
        val_df: Validation data (pre-split)
        model_type: Type of model to use ('lgbm' or 'catboost')
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (-1 for all available CPUs)
        pruner: Pruning strategy - 'tpg' (Tree-Structured Pruning) or 'median'
        sampler: Sampler - 'tpes' (Tree-structured Parzen Estimator) or 'hbopt' (Hyperband)
        verbose: Verbosity level (0-3)

    Returns:
        dict: Results containing best hyperparameters and metrics
    """
    print(f"Total rows: {len(train_df) + len(val_df)}")
    print(f"Train size: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)):.2%})")
    print(f"Val size: {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)):.2%})")

    # Create Optuna study
    study = create_optuna_study(
        study_name='f1_strategy_optuna',
        n_trials=n_trials,
        n_jobs=n_jobs,
        pruner=pruner,
        sampler=sampler
    )

    # Run optimization
    print(f"\nStarting Optuna Search ({n_trials} trials)...")
    study.optimize(
        lambda trial: objective(trial, train_df, val_df, model_type=model_type, n_trials=n_trials, n_jobs=n_jobs),
        n_trials=n_trials,
        show_progress_bar=verbose > 0
    )

    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = -study.best_value  # Convert back to positive AUC-ROC

    # Retrain best model
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    X_train, y_train, imputer, encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=False)

    if model_type == 'lgbm':
        model = lgb.LGBMClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            learning_rate=best_params.get('learning_rate', 0.1),
            num_leaves=best_params.get('num_leaves', 31),
            max_depth=best_params.get('max_depth', 10),
            min_child_samples=best_params.get('min_child_samples', 20),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            reg_alpha=best_params.get('reg_alpha', 0.1),
            reg_lambda=best_params.get('reg_lambda', 0.1),
            min_data_for_leaf=best_params.get('min_data_for_leaf', 10),
            min_data_for_host=best_params.get('min_data_for_host', 2),
            feature_fraction=best_params.get('feature_fraction', 0.8),
            bagging_fraction=best_params.get('bagging_fraction', 0.8),
            bagging_freq=best_params.get('bagging_freq', 1),
            verbose=-1,
            random_state=best_params.get('seed', 42),
            n_jobs=n_jobs
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    else:
        model = CatBoostClassifier(
            iterations=best_params.get('iterations', 100),
            learning_rate=best_params.get('learning_rate', 0.1),
            depth=best_params.get('depth', 6),
            l2_leaf_reg=best_params.get('l2_leaf_reg', 1.0),
            random_seed=best_params.get('random_seed', 42),
            loss_function=best_params.get('loss_function', 'Logloss'),
            verbose=0,
            random_state=best_params.get('random_seed', 42),
            thread_count=n_jobs
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)

    # Final evaluation
    y_pred_val = np.asarray(model.predict(X_val))
    probs = np.asarray(model.predict_proba(X_val))
    y_pred_proba_val = probs[:, 1] if hasattr(probs, 'toarray') else probs[:, 1]

    val_precision = precision_score(y_val, y_pred_val)
    val_auc_roc = roc_auc_score(y_val, y_pred_proba_val)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "val_precision": val_precision,
        "val_auc_roc": val_auc_roc,
        "best_model": model,
        "study": study
    }

