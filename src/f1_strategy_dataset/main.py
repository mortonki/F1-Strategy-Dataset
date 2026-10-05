import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_score
import lightgbm as lgb
from f1_strategy_dataset.preprocessing import preprocess_f1_data
from f1_strategy_dataset.mlflow_utils import init_experiment, log_parameters, log_metrics, log_model, log_experiment_info, log_data_stats, log_model_info, log_all
from f1_strategy_dataset.hyperparameter_search_optuna import run_optuna_search


def main():
    # Parse command-line arguments for hyperparameter tuning
    parser = argparse.ArgumentParser(
        description='F1 Strategy Prediction using LightGBM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--search',
        type=str,
        default='train',
        choices=['train', 'random'],
        help='Mode: train (single model), random (random search)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.05,
        help='Learning rate (step-down factor) for the model'
    )
    parser.add_argument(
        '--num_leaves',
        type=int,
        default=31,
        help='Maximum number of leaves in each tree'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all available CPUs)'
    )
    
    args = parser.parse_args()
    
    # Load the F1 Strategy Dataset
    df = pd.read_csv('data/f1_strategy_dataset_v4.csv')
    
    # Time-series aware train/test split
    train_df = df[df['Year'].isin([2022, 2023])].copy()
    val_df = df[df['Year'] == 2024].copy()
    test_df = df[df['Year'] == 2025].copy()
    
    print(f"Total rows: {len(df)}")
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Val size: {len(val_df)} ({len(val_df)/len(df):.2%})")
    print(f"Test size: {len(test_df)} ({len(test_df)/len(df):.2%})")
    
    # Create imputer and encoder
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    
    # Preprocess training data
    X_train, y_train, imputer, encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    
    # Preprocess validation data
    X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=False)
    
    # Preprocess test data
    #X_test, y_test = preprocess_f1_data(test_df, imputer, encoder, is_training=False)
    
    # Run Optuna search for optimal hyperparameters
    print("\n=== Running Optuna Search ===")
    results = run_optuna_search(df, n_trials=50, n_jobs=args.n_jobs, verbose=1)
    
    # Get best model by retraining with best parameters
    best_params = results['best_params']
    
    # Define LightGBM classifier
    best_model = lgb.LGBMClassifier(
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
        random_state=best_params['seed'],
        n_jobs=args.n_jobs
    )
    
    # Train final model with best parameters
    best_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_val = np.array(best_model.predict(X_val))
    y_pred_proba_val = best_model.predict_proba(X_val)
    toarray = getattr(y_pred_proba_val, 'toarray', None)
    if callable(toarray):
        y_pred_proba_val = np.asarray(toarray())
    else:
        y_pred_proba_val = np.asarray(y_pred_proba_val)

    print("\n=== Final Validation Results ===")
    precision = precision_score(y_val, y_pred_val)
    print(f"Precision: {precision:.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_val, y_pred_proba_val[:, 1]):.4f}")
    
    # Log best model
    log_all(best_model, type('Args', (), {
        'n_estimators': best_params['n_estimators'],
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves'],
        'max_depth': best_params['max_depth'],
        'min_child_samples': best_params['min_child_samples'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'min_data_for_leaf': best_params['min_data_for_leaf'],
        'min_data_for_host': best_params['min_data_for_host'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq': best_params['bagging_freq'],
        'verbose': best_params['verbose'],
        'seed': best_params['seed'],
        'random_state': best_params['seed'],
        'n_jobs': args.n_jobs
    }), {
        "val_precision": precision_score(y_val, y_pred_val),
        "val_auc_roc": roc_auc_score(y_val, y_pred_proba_val[:, 1])
    })
    
    # Save best model
    log_model(best_model, name="best_model")


if __name__ == "__main__":
    main()
