"""
Hyperparameter Search Module for F1 Strategy Prediction

This module provides functions for hyperparameter tuning using RandomizedSearchCV.
It integrates with MLflow for tracking experiments and logging results.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from preprocessing import preprocess_f1_data
from mlflow_utils import init_experiment, log_all, log_model


def run_random_search(
    df: pd.DataFrame,
    n_iter: int = 20,
    n_jobs: int = -1,
    verbose: int = 1
) -> dict:
    """
    Run RandomizedSearchCV for hyperparameter tuning.
    
    Args:
        df: Full dataset with F1 race data
        n_iter: Number of iterations to run
        n_jobs: Number of parallel jobs (-1 for all available CPUs)
        verbose: Verbosity level (0-3)
    
    Returns:
        dict: Results containing best hyperparameters and metrics
    """
    # Load the F1 Strategy Dataset
    train_df = df[df['Year'].isin([2022, 2023])].copy()
    val_df = df[df['Year'] == 2024].copy()
    
    print(f"Total rows: {len(df)}")
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Val size: {len(val_df)} ({len(val_df)/len(df):.2%})")
    
    # Define hyperparameter grid
    param_grids = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [10, 20, 31, 50, 100],
        'max_depth': [None, 3, 5, 7, 9, 11],
        'min_child_samples': [10, 20, 50, 100],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.0, 0.1, 0.5, 1.0],
        'min_data_for_leaf': [5, 10, 20],
        'min_data_for_host': [1, 2, 4],
        'feature_fraction': [0.7, 0.8, 1.0],
        'bagging_fraction': [0.7, 0.8, 1.0],
        'bagging_freq': [0, 1, 3],
        'verbose': [-1, 0, 1],
        'seed': [42, 123, 456]
    }
    
    # Create imputer and encoder
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    
    # Preprocess training data
    X_train, y_train, imputer, encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    
    # Preprocess validation data
    X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=False)
    
    # Define LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        n_jobs=n_jobs,
        verbose=0,  # Suppress warnings during hyperparameter search
        random_state=42
    )
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_grids,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=0,  # Suppress progress messages during random search
        refit=True
    )
    
    # Run random search
    print(f"\nStarting Random Search ({n_iter} iterations)...")
    random_search.fit(X_train, y_train)
    
    # Get best results
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    # Evaluate on validation set
    best_model = random_search.best_estimator_
    y_pred_val = best_model.predict(X_val)
    y_pred_proba_val = best_model.predict_proba(X_val)
    
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_auc_roc = roc_auc_score(y_val, y_pred_proba_val[:, 1])
    
    # Log to MLflow
    print("\n=== Random Search Results ===")
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score (AUC-ROC): {best_score:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC-ROC: {val_auc_roc:.4f}")
    
    # Log to MLflow
    log_all(best_model, type('Args', (), {
        'n_estimators': best_params.get('n_estimators', 100),
        'learning_rate': best_params.get('learning_rate', 0.05),
        'num_leaves': best_params.get('num_leaves', 31),
        'max_depth': best_params.get('max_depth', None),
        'min_child_samples': best_params.get('min_child_samples', 10),
        'subsample': best_params.get('subsample', 1.0),
        'colsample_bytree': best_params.get('colsample_bytree', 1.0),
        'reg_alpha': best_params.get('reg_alpha', 0.0),
        'reg_lambda': best_params.get('reg_lambda', 0.0),
        'min_data_for_leaf': best_params.get('min_data_for_leaf', 5),
        'min_data_for_host': best_params.get('min_data_for_host', 1),
        'feature_fraction': best_params.get('feature_fraction', 1.0),
        'bagging_fraction': best_params.get('bagging_fraction', 1.0),
        'bagging_freq': best_params.get('bagging_freq', 0),
        'verbose': best_params.get('verbose', 0),
        'seed': best_params.get('seed', 42),
        'random_state': best_params.get('random_state', 42),
        'n_jobs': best_params.get('n_jobs', -1)
    }), {
        "val_accuracy": val_accuracy,
        "val_auc_roc": val_auc_roc,
        "cv_auc_roc": best_score
    })
    
    # Save best model
    log_model(best_model, artifact_path="best_model")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'val_accuracy': val_accuracy,
        'val_auc_roc': val_auc_roc,
        'best_model': best_model
    }


def main():
    """Main function for hyperparameter search."""
    parser = argparse.ArgumentParser(
        description='F1 Strategy Hyperparameter Search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n_iter',
        type=int,
        default=20,
        help='Number of iterations for random search'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all available CPUs)'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0-3)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv('f1_strategy_dataset_v4.csv')
    
    # Run random search
    results = run_random_search(df, n_iter=args.n_iter, n_jobs=args.n_jobs, verbose=args.verbose)
    
    print("\n=== Final Results ===")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best CV Score (AUC-ROC): {results['best_score']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    print(f"Validation AUC-ROC: {results['val_auc_roc']:.4f}")
    
    # Print best hyperparameters
    print("\n=== Best Hyperparameters ===")
    for name, value in results['best_params'].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()