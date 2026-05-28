import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from preprocessing import preprocess_f1_data
from mlflow_utils import init_experiment, log_parameters, log_metrics, log_model, log_experiment_info, log_data_stats, log_model_info, log_all
from hyperparameter_search import run_random_search


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
        '--n_estimators',
        type=int,
        default=1000,
        help='Number of trees in the LightGBM model'
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
    parser.add_argument(
        '--n_iter',
        type=int,
        default=20,
        help='Number of iterations for random search'
    )
    
    args = parser.parse_args()
    
    # Load the F1 Strategy Dataset
    df = pd.read_csv('f1_strategy_dataset_v4.csv')
    
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
    
    if args.search == 'train':
        # Train single model with specified hyperparameters
        model = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            random_state=args.random_state,
            n_jobs=args.n_jobs
        )
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)
        
        print("\n=== Training Results ===")
        print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_train, y_pred_proba_train[:, 1]):.4f}")

        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        y_pred_proba_val = model.predict_proba(X_val)
        
        print("\n=== Validation Results ===")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_val, y_pred_proba_val[:, 1]):.4f}")
        
        # Evaluate on test set
        #y_pred_test = model.predict(X_test)
        #y_pred_proba_test = model.predict_proba(X_test)
        
        #print("\n=== Test Results ===")
        #print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        #print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba_test[:, 1]):.4f}")
        
        # Log only essential hyperparameters and metrics to MLflow
        # Reduced scope: no dataset statistics, class distributions, or feature names
        log_all(model, args, {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_auc_roc": roc_auc_score(y_train, y_pred_proba_train[:, 1]),
            "val_accuracy": accuracy_score(y_val, y_pred_val),
            "val_auc_roc": roc_auc_score(y_val, y_pred_proba_val[:, 1])
        })
        
    elif args.search == 'random':
        # Run random search
        print("\n=== Running Random Search ===")
        results = run_random_search(df, n_iter=args.n_iter, n_jobs=args.n_jobs, verbose=1)
        
        print("\n=== Best Hyperparameters ===")
        print(f"n_estimators: {results['best_params']['n_estimators']}")
        print(f"learning_rate: {results['best_params']['learning_rate']}")
        print(f"num_leaves: {results['best_params']['num_leaves']}")
        print(f"max_depth: {results['best_params']['max_depth']}")
        print(f"min_child_samples: {results['best_params']['min_child_samples']}")
        print(f"subsample: {results['best_params']['subsample']}")
        print(f"colsample_bytree: {results['best_params']['colsample_bytree']}")
        print(f"reg_alpha: {results['best_params']['reg_alpha']}")
        print(f"reg_lambda: {results['best_params']['reg_lambda']}")
        print(f"min_data_for_leaf: {results['best_params']['min_data_for_leaf']}")
        print(f"min_data_for_host: {results['best_params']['min_data_for_host']}")
        print(f"feature_fraction: {results['best_params']['feature_fraction']}")
        print(f"bagging_fraction: {results['best_params']['bagging_fraction']}")
        print(f"bagging_freq: {results['best_params']['bagging_freq']}")
        print(f"verbose: {results['best_params']['verbose']}")
        print(f"seed: {results['best_params']['seed']}")
        
        print("\n=== Best CV Score ===")
        print(f"AUC-ROC: {results['best_score']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation AUC-ROC: {results['val_auc_roc']:.4f}")
        
        # Train final model with best parameters
        best_model = lgb.LGBMClassifier(
            n_estimators=results['best_params']['n_estimators'],
            learning_rate=results['best_params']['learning_rate'],
            num_leaves=results['best_params']['num_leaves'],
            max_depth=results['best_params']['max_depth'],
            min_child_samples=results['best_params']['min_child_samples'],
            subsample=results['best_params']['subsample'],
            colsample_bytree=results['best_params']['colsample_bytree'],
            reg_alpha=results['best_params']['reg_alpha'],
            reg_lambda=results['best_params']['reg_lambda'],
            min_data_for_leaf=results['best_params']['min_data_for_leaf'],
            min_data_for_host=results['best_params']['min_data_for_host'],
            feature_fraction=results['best_params']['feature_fraction'],
            bagging_fraction=results['best_params']['bagging_fraction'],
            bagging_freq=results['best_params']['bagging_freq'],
            verbose=results['best_params']['verbose'],
            random_state=results['best_params']['seed'],
            n_jobs=args.n_jobs
        )
        best_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = best_model.predict(X_val)
        y_pred_proba_val = best_model.predict_proba(X_val)
        
        print("\n=== Final Validation Results ===")
        print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_val, y_pred_proba_val[:, 1]):.4f}")
        
        # Log best model
        log_all(best_model, type('Args', (), {
            'n_estimators': results['best_params']['n_estimators'],
            'learning_rate': results['best_params']['learning_rate'],
            'num_leaves': results['best_params']['num_leaves'],
            'max_depth': results['best_params']['max_depth'],
            'min_child_samples': results['best_params']['min_child_samples'],
            'subsample': results['best_params']['subsample'],
            'colsample_bytree': results['best_params']['colsample_bytree'],
            'reg_alpha': results['best_params']['reg_alpha'],
            'reg_lambda': results['best_params']['reg_lambda'],
            'min_data_for_leaf': results['best_params']['min_data_for_leaf'],
            'min_data_for_host': results['best_params']['min_data_for_host'],
            'feature_fraction': results['best_params']['feature_fraction'],
            'bagging_fraction': results['best_params']['bagging_fraction'],
            'bagging_freq': results['best_params']['bagging_freq'],
            'verbose': results['best_params']['verbose'],
                'seed': results['best_params']['seed'],
                'random_state': results['best_params'].get('seed', 42),
                'n_jobs': results['best_params'].get('n_jobs', -1)
            }), {
            "val_accuracy": accuracy_score(y_val, y_pred_val),
            "val_auc_roc": roc_auc_score(y_val, y_pred_proba_val[:, 1])
        })
        
        # Save best model
        log_model(best_model, artifact_path="best_model")


if __name__ == "__main__":
    main()