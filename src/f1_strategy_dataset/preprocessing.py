"""
F1 Strategy Dataset Preprocessing Module

This module contains functions for preprocessing F1 race data, including:
- Creating lagged features for Position, pace-change (LapTime_Delta), and degradation metrics
- Encoding categorical variables (Compound)
- Handling missing values
"""

import pandas as pd
import numpy as np
import scipy.sparse
from typing import cast
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def preprocess_f1_data(
    df: pd.DataFrame,
    imputer: SimpleImputer,
    encoder: OneHotEncoder,
    is_training: bool = False
) -> tuple:
    """
    Preprocess F1 data by creating lagged features and encoding categorical variables.
    
    This function applies the same transformations to train, validation, and test datasets.
    It uses the provided imputer and encoder to ensure consistent preprocessing across
    all datasets.
    
    Args:
        df: Input DataFrame with F1 race data (should contain columns: Position, 
            LapTime_Delta, Cumulative_Degradation, Compound, PitNextLap)
        imputer: Trained SimpleImputer for handling missing values
        encoder: Trained OneHotEncoder for Compound and Year columns
        is_training: If True, returns X and y separately; otherwise returns processed DataFrame
    
    Returns:
        If is_training=True: Tuple of (X, y) where X is the feature DataFrame and y is the target
        If is_training=False: Processed DataFrame with all features
    
    Example:
        >>> imputer = SimpleImputer(strategy='most_frequent')
        >>> encoder = OneHotEncoder(handle_unknown='ignore')
        >>> X_train, y_train = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
        >>> X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=True)
        >>> X_test, y_test = preprocess_f1_data(test_df, imputer, encoder, is_training=True)
    """
    # Create lagged features for Position, pace-change (LapTime_Delta), and degradation metrics
    # Group by Driver, Race, Year to maintain temporal order within sequences
    df['Position_lag1'] = df.groupby(['Year', 'Race', 'Driver'])['Position'].shift(1)
    df['Position_lag2'] = df.groupby(['Year', 'Race', 'Driver'])['Position'].shift(2)
    df['Position_lag3'] = df.groupby(['Year', 'Race', 'Driver'])['Position'].shift(3)
    
    # LapTime_Delta lags (pace change over time, more relevant for pit decisions than
    # absolute pace. Raw LapTime (s) is dropped as it is highly collinear with LapTime_Delta).
    df['LapTime_Delta_lag1'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime_Delta'].shift(1)
    df['LapTime_Delta_lag2'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime_Delta'].shift(2)
    df['LapTime_Delta_lag3'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime_Delta'].shift(3)
    
    # Rolling average pace (using LapTime_Delta as a proxy for pace change)
    # Rolling average of LapTime_Delta over 3 laps
    df['Rolling_avg_pace'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime_Delta'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Rolling degradation (rolling average of Cumulative_Degradation)
    df['Rolling_degradation'] = df.groupby(['Year', 'Race', 'Driver'])['Cumulative_Degradation'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # TyreLife is the strongest predictor (corr ~0.266) but was previously dropped.
    # Add lagged/rolling tyre-life features (raw TyreLife/Normalized_TyreLife are
    # already present in the data and added to feature_cols in main.py).
    df['TyreLife_lag1'] = df.groupby(['Year', 'Race', 'Driver'])['TyreLife'].shift(1)
    df['TyreLife_lag2'] = df.groupby(['Year', 'Race', 'Driver'])['TyreLife'].shift(2)

    # Rolling average of TyreLife (tyre degradation rate)
    df['Rolling_avg_TyreLife'] = df.groupby(['Year', 'Race', 'Driver'])['TyreLife'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Fill NaN values (from shifts) with 0
    # Only fill numeric columns with 0, leave string/categorical columns unchanged
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Drop Driver and Race columns (they're used for grouping but not needed as features)
    df = df.drop(['Driver', 'Race'], axis=1)
    
    # Impute missing values in Compound column.
    compound_values = df['Compound'].astype(object).to_numpy().reshape(-1, 1)
    if is_training:
        compound_imputed = np.asarray(imputer.fit_transform(compound_values))
    else:
        compound_imputed = np.asarray(imputer.transform(compound_values))
    df['Compound'] = compound_imputed[:, 0]

    # One-Hot Encode Compound and Year
    # Ensure they are treated as strings for OneHotEncoder
    df['Compound'] = df['Compound'].astype(str)
    df['Year'] = df['Year'].astype(str)
    
    cols_to_encode = ['Compound', 'Year']
    
    if is_training:
        encoded_data = encoder.fit_transform(df[cols_to_encode])
    else:
        encoded_data = encoder.transform(df[cols_to_encode])
    
    # Convert to dense array if necessary
    if scipy.sparse.issparse(encoded_data):
        encoded_data = cast(scipy.sparse.csr_matrix, encoded_data).toarray()
    else:
        encoded_data = np.asarray(encoded_data)
    encoded_data = cast(np.ndarray, encoded_data)
        
    encoded_cols = encoder.get_feature_names_out(cols_to_encode)
    
    # Create a DataFrame from the encoded results and concatenate
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Drop the original Compound, Year, and raw LapTime (s) columns
    df.drop(columns=cols_to_encode + ['LapTime (s)'], inplace=True)
    
    # Separate features and target variable
    X = df.drop('PitNextLap', axis=1)
    y = df['PitNextLap']

    if is_training:     
        return X, y, imputer, encoder
    else:
        return X, y


def create_lagged_features(
    df: pd.DataFrame,
    group_cols: list = ['Driver', 'Race', 'Year']
) -> pd.DataFrame:
    """
    Create lagged features for Position, pace-change (LapTime_Delta), and degradation metrics.
    
    This function creates lagged features grouped by the specified columns.
    It can be used as a standalone function or as part of the preprocessing pipeline.
    
    Args:
        df: Input DataFrame with F1 race data
        group_cols: List of columns to group by for maintaining temporal order
    
    Returns:
        DataFrame with lagged features added
    """
    # Create lagged features for Position
    df['Position_lag1'] = df.groupby(group_cols)['Position'].shift(1)
    df['Position_lag2'] = df.groupby(group_cols)['Position'].shift(2)
    df['Position_lag3'] = df.groupby(group_cols)['Position'].shift(3)
    
    # LapTime_Delta lags (pace change over time, more relevant for pit decisions than
    # absolute pace. Raw LapTime (s) is dropped as it is highly collinear with LapTime_Delta).
    df['LapTime_Delta_lag1'] = df.groupby(group_cols)['LapTime_Delta'].shift(1)
    df['LapTime_Delta_lag2'] = df.groupby(group_cols)['LapTime_Delta'].shift(2)
    df['LapTime_Delta_lag3'] = df.groupby(group_cols)['LapTime_Delta'].shift(3)
    
    # Rolling average pace (using LapTime_Delta as a proxy for pace change)
    df['Rolling_avg_pace'] = df.groupby(group_cols)['LapTime_Delta'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Rolling degradation (rolling average of Cumulative_Degradation)
    df['Rolling_degradation'] = df.groupby(group_cols)['Cumulative_Degradation'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # TyreLife is the strongest predictor (corr ~0.266) but was previously dropped.
    # Add lagged/rolling tyre-life features (raw TyreLife/Normalized_TyreLife are
    # already present in the data and added to feature_cols in main.py).
    df['TyreLife_lag1'] = df.groupby(group_cols)['TyreLife'].shift(1)
    df['TyreLife_lag2'] = df.groupby(group_cols)['TyreLife'].shift(2)

    # Rolling average of TyreLife (tyre degradation rate)
    df['Rolling_avg_TyreLife'] = df.groupby(group_cols)['TyreLife'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Fill NaN values (from shifts) with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df