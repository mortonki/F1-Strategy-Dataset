import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from preprocessing import preprocess_f1_data


def main():
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
    X_test, y_test = preprocess_f1_data(test_df, imputer, encoder, is_training=False)
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
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


if __name__ == "__main__":
    main()
