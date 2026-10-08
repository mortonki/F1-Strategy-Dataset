import mlflow
from mlflow import sklearn
import argparse
import pandas as pd
import numpy as np
import json
import yaml
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_score
import lightgbm as lgb
from f1_strategy_dataset.preprocessing import preprocess_f1_data
from f1_strategy_dataset.hyperparameter_search_optuna import run_optuna_search
from f1_strategy_dataset import settings

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_best_params(params, path):
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)

def load_best_params(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_data(data_path):
    df = pd.read_csv(data_path)
    # Time-series aware train/test split
    train_df = df[df['Year'].isin([2022, 2023])].copy()
    val_df = df[df['Year'] == 2024].copy()
    test_df = df[df['Year'] == 2025].copy()
    return df, train_df, val_df, test_df

def prepare_data(train_df, val_df):
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    
    X_train, y_train, imputer, encoder = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
    X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=False)
    
    return X_train, y_train, X_val, y_val, imputer, encoder

def run_tuning(train_df, val_df, config):
    print("\n=== Running Optuna Search ===")
    # Pass both train_df and val_df to allow the tuner to handle internal splits correctly
    results = run_optuna_search(
        train_df, 
        val_df,
        n_trials=config['OPTUNA_TRIALS'], 
        n_jobs=config['DEFAULT_PARAMS']['n_jobs'], 
        verbose=1
    )
    
    best_params = results['best_params']
    save_best_params(best_params, config['PARAMS_PATH'])
    print(f"Best parameters saved to {config['PARAMS_PATH']}")
    return best_params

def train_and_evaluate(X_train, y_train, X_val, y_val, best_params, config):
    print("\n=== Training Final Model ===")
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
        n_jobs=config['DEFAULT_PARAMS']['n_jobs']
    )
    
    best_model.fit(X_train, y_train)
    
    y_pred_val = np.array(best_model.predict(X_val))
    y_pred_proba_val = best_model.predict_proba(X_val)
    toarray = getattr(y_pred_proba_val, 'toarray', None)
    if callable(toarray):
        y_pred_proba_val = np.asarray(toarray())
    else:
        y_pred_proba_val = np.asarray(y_pred_proba_val)

    print("\n=== Final Validation Results ===")
    precision = precision_score(y_val, y_pred_val)
    auc_roc = roc_auc_score(y_val, y_pred_proba_val[:, 1])
    print(f"Precision: {precision:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    with mlflow.start_run(nested=True):
        for key, value in best_params.items():
            if value is not None:
                mlflow.log_param(key, str(value))
        
        metrics = {
            "val_precision": precision,
            "val_auc_roc": auc_roc
        }
        mlflow.log_metrics(metrics)
        sklearn.log_model(best_model, name="model")
    
    return best_model

def main():
    parser = argparse.ArgumentParser(
        description='F1 Strategy Prediction using LightGBM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='train', choices=['tune', 'train', 'evaluate'],
                        help='Execution mode: tune (Optuna), train (Fit with saved params), evaluate (Inference)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    settings.init_mlflow(config)

    df, train_df, val_df, test_df = load_data(config['DATA_PATH'])
    
    print(f"Total rows: {len(df)}")
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Val size: {len(val_df)} ({len(val_df)/len(df):.2%})")
    print(f"Test size: {len(test_df)} ({len(test_df)/len(df):.2%})")
    
    X_train, y_train, X_val, y_val, imputer, encoder = prepare_data(train_df, val_df)
    
    if args.mode == 'tune':
        run_tuning(train_df, val_df, config)
    elif args.mode == 'train':
        best_params = load_best_params(config['PARAMS_PATH'])
        if best_params is None:
            print(f"Error: No best parameters found at {config['PARAMS_PATH']}. Please run in 'tune' mode first.")
            return
        train_and_evaluate(X_train, y_train, X_val, y_val, best_params, config)
    elif args.mode == 'evaluate':
        best_params = load_best_params(config['PARAMS_PATH'])
        if best_params is None:
            print(f"Error: No best parameters found at {config['PARAMS_PATH']}. Please run in 'tune' mode first.")
            return
        # For evaluation, we could also load a serialized model instead of retraining
        # But for now, let's just retrain and show results as requested
        train_and_evaluate(X_train, y_train, X_val, y_val, best_params, config)

if __name__ == "__main__":
    main()
