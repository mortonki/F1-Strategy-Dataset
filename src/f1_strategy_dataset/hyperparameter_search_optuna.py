"""
Hyperparameter Search Module for F1 Strategy Prediction using Optuna

This module provides functions for hyperparameter tuning using Optuna with Bayesian optimization.
Optuna is memory-efficient and state-of-the-art for hyperparameter tuning.
"""

import argparse
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from f1_strategy_dataset.preprocessing import preprocess_f1_data
from f1_strategy_dataset.mlflow_utils import init_experiment, log_all, log_model

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
    study: optuna.Study,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str = 'lgbm',
    n_trials: int = 50,
    n_jobs: int = -1
):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        study: Optuna study object
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
        n_estimators = study.suggest_int('n_estimators', low=100, high=1000)
        learning_rate = study.suggest_float('learning_rate', low=0.01, high=0.2, log=True)
        num_leaves = study.suggest_int('num_leaves', low=10, high=100)
        max_depth = study.suggest_int('max_depth', low=3, high=11)
        min_child_samples = study.suggest_int('min_child_samples', low=10, high=100)
        subsample = study.suggest_float('subsample', low=0.8, high=1.0)
        colsample_bytree = study.suggest_float('colsample_bytree', low=0.8, high=1.0)
        reg_alpha = study.suggest_float('reg_alpha', low=0.0, high=1.0)
        reg_lambda = study.suggest_float('reg_lambda', low=0.0, high=1.0)
        min_data_for_leaf = study.suggest_int('min_data_for_leaf', low=5, high=20)
        min_data_for_host = study.suggest_int('min_data_for_host', low=1, high=4)
        feature_fraction = study.suggest_float('feature_fraction', low=0.7, high=1.0)
        bagging_fraction = study.suggest_float('bagging_fraction', low=0.7, high=1.0)
        bagging_freq = study.suggest_int('bagging_freq', low=0, high=3)
        verbose = study.suggest_int('verbose', low=-1, high=1)
        seed = study.suggest_int('seed', low=42, high=123)

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
        iterations = study.suggest_int('iterations', low=100, high=1000)
        learning_rate = study.suggest_float('learning_rate', low=0.01, high=0.2, log=True)
        depth = study.suggest_int('depth', low=3, high=11)
        l2_leaf_reg = study.suggest_float('l2_leaf_reg', low=1e-3, high=10.0, log=True)
        random_seed = study.suggest_int('random_seed', low=42, high=123)
        loss_function = study.suggest_categorical('loss_function', ['Logloss', 'AUC'])
        
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
    y_pred_val = model.predict(X_val)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1] # Ensure probability is used for AUC
    
    val_precision = precision_score(y_val, y_pred_val)
    val_auc_roc = roc_auc_score(y_val, y_pred_proba_val)
    
    # Return negative AUC-ROC (Optuna minimizes)
    return -val_auc_roc

def run_optuna_search(
    df: pd.DataFrame,
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
        df: Full dataset with F1 race data
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (-1 for all available CPUs)
        pruner: Pruning strategy - 'tpg' or 'median'
        sampler: Sampler - 'tpes' or 'hbopt'
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
        lambda study: objective(study, train_df, val_df, n_trials=n_trials, n_jobs=n_jobs),
        n_trials=n_trials,
        show_progress_bar=verbose > 0
    )
    
    # Get best results
    best_trial = study.best_trial
    # Use the best trial parameters directly
    best_params = best_trial.params
    best_score = -study.best_value  # Convert back to positive AUC-ROC
    
    # Get best model by retraining with best parameters
    best_params = study.best_trial.params
    
    # Create imputer and encoder
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    
    # Preprocess training data
    X_train, y_train, imputer, encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    
    # Preprocess validation data
    X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=False)
    
    # Define LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        num_leaves=best_params['num_leaves'],
        max_depth=best_params['max_depth'],
        min_child_samples=best_params['min_child_samples'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        min_data_for_leaf=best_params['min_data_for_leaf'],
        min_data_for_host=best_params['min_data_for_host'],
        feature_fraction=best_params['feature_fraction'],
        bagging_fraction=best_params['bagging_fraction'],
        bagging_freq=best_params['bagging_freq'],
        verbose=best_params['verbose'],
        n_jobs=n_jobs,
        random_state=best_params['seed']
    )
    
    # Train model with early stopping
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Validation data for early stopping
        callbacks=[lgb.early_stopping(stopping_rounds=50)], # Number of early stopping rounds
    )
    best_model = lgb_model
    
    # Evaluate on validation set
    y_pred_val = best_model.predict(X_val)
    y_pred_proba_val = best_model.predict_proba(X_val)

    val_precision = precision_score(y_val, y_pred_val)
    val_auc_roc = roc_auc_score(y_val, y_pred_proba_val[:, 1])
    
    # Log to MLflow
    print("\n=== Optuna Search Results ===")
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score (AUC-ROC): {best_score:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation AUC-ROC: {val_auc_roc:.4f}")
    
    # Log to MLflow
    log_all(best_model, {
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
        'n_jobs': n_jobs
    }, {
        "val_precision": val_precision,
        "val_auc_roc": val_auc_roc,
        "cv_auc_roc": best_score
    })
    
    # Save best model
    log_model(best_model, name="best_model")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'val_precision': val_precision,
        'val_auc_roc': val_auc_roc,
        'best_model': best_model,
        'study': study
    }


def main():
    """Main function for Optuna hyperparameter search."""
    parser = argparse.ArgumentParser(
        description='F1 Strategy Hyperparameter Search using Optuna',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of trials to run'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all available CPUs)'
    )
    parser.add_argument(
        '--pruner',
        type=str,
        default='tpg',
        choices=['tpg', 'median'],
        help='Pruning strategy'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='tpes',
        choices=['tpes', 'hbopt'],
        help='Sampler - TPES (Tree-structured Parzen Estimator) or HBOpt (Hyperband)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='lgbm',
        choices=['lgbm', 'catboost'],
        help='Type of model to use for hyperparameter search (lgbm or catboost)'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0-3)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv('data/f1_strategy_dataset_v4.csv')
    
    # Run search
    results = run_optuna_search(
        df,
        model_type=args.model_type,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        pruner=args.pruner,
        sampler=args.sampler,
        verbose=args.verbose
    )
    
    print("\n=== Final Results ===")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best CV Score (AUC-ROC): {results['best_score']:.4f}")
    print(f"Validation Precision: {results['val_precision']:.4f}")
    print(f"Validation AUC-ROC: {results['val_auc_roc']:.4f}")
    
    # Print all trial results
    #print("\n=== All Trial Results ===")
    #for i, trial in enumerate(results['study'].trials):
    #    print(f"Trial {i+1}: {trial.value:.4f} (AUC-ROC: {-trial.value:.4f})")


if __name__ == "__main__":
    main()