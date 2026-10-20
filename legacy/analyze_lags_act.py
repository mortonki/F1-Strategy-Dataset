import pandas as pd
import numpy as np
import json
import os
import sys

# Add current directory to path for local imports
sys.path.append(os.getcwd())

try:
    from f1_strategy_dataset.preprocessing import preprocess_f1_data
except ImportError:
    # Fallback if the package structure isn't recognized directly
    import sys
    sys.path.append(os.path.abspath("src"))
    from f1_strategy_dataset.preprocessing import preprocess_f1_data

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def main():
    # Load the dataset
    df = pd.read_csv('data/f1_strategy_dataset_v4.csv')
    
    # Initialize transformers (mimicking the pipeline)
    imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    
    # Preprocess the data
    # Note: preprocess_f1_data returns X, y, imputer, encoder if is_training=True
    X, y, _, _ = preprocess_f1_data(df, imputer, encoder, is_training=True)
    
    # Now X contains all the engineered features including lags
    # Let's identify the lag features specifically
    lag_features = [
        'Position_lag1', 'Position_lag2', 'Position_lag3',
        'LapTime_Delta_lag1', 'LapTime_Delta_lag2', 'LapTime_Delta_lag3',
        'TyreLife_lag1', 'TyreLife_lag2',
        'Rolling_avg_pace', 'Rolling_degradation', 'Rolling_avg_TyreLife'
    ]
    
    # Filter to only those that actually exist in X
    existing_lags = [f for f in lag_features if f in X.columns]
    
    correlations = {}
    
    # Correlation with target y
    for f in existing_lags:
        correlations[f] = X[f].corr(y)
        
    # Also include some base features for comparison
    base_features = ['Position', 'LapTime_Delta', 'Cumulative_Degradation', 'TyreLife']
    for f in base_features:
        if f in X.columns:
            correlations[f] = X[f].corr(y)
            
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("--- Correlation Analysis ---")
    for name, corr in sorted_correlations:
        print(f"{name}: {corr:.4f}")
        
    with open('correlation_results.json', 'w') as f:
        json.dump(dict(sorted_correlations), f, indent=4)

if __name__ == "__main__":
    main()
