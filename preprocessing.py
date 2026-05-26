"""
F1 Strategy Dataset Preprocessing Module

This module contains functions for preprocessing F1 race data, including:
- Creating lagged features for Position, LapTime, and degradation metrics
- Encoding categorical variables (Compound)
- Handling missing values
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def preprocess_f1_data(
    df: pd.DataFrame,
    imputer: SimpleImputer,
    encoder: OrdinalEncoder,
    is_training: bool = False
) -> tuple:
    """
    Preprocess F1 data by creating lagged features and encoding categorical variables.
    
    This function applies the same transformations to train, validation, and test datasets.
    It uses the provided imputer and encoder to ensure consistent preprocessing across
    all datasets.
    
    Args:
        df: Input DataFrame with F1 race data (should contain columns: Position, 
            LapTime (s), LapTime_Delta, Cumulative_Degradation, Compound, PitNextLap)
        imputer: Trained SimpleImputer for handling missing values
        encoder: Trained OrdinalEncoder for Compound column
        is_training: If True, returns X and y separately; otherwise returns processed DataFrame
    
    Returns:
        If is_training=True: Tuple of (X, y) where X is the feature DataFrame and y is the target
        If is_training=False: Processed DataFrame with all features
    
    Example:
        >>> imputer = SimpleImputer(strategy='most_frequent')
        >>> encoder = OrdinalEncoder()
        >>> X_train, y_train = preprocess_f1_data(train_df, imputer, encoder, is_training=True)
        >>> X_val, y_val = preprocess_f1_data(val_df, imputer, encoder, is_training=True)
        >>> X_test, y_test = preprocess_f1_data(test_df, imputer, encoder, is_training=True)
    """
    # Create lagged features for Position, LapTime, and degradation metrics
    # Group by Driver, Race, Year to maintain temporal order within sequences
    df['Position_lag1'] = df.groupby(['Driver', 'Race', 'Year'])['Position'].shift(1)
    df['Position_lag2'] = df.groupby(['Driver', 'Race', 'Year'])['Position'].shift(2)
    df['Position_lag3'] = df.groupby(['Driver', 'Race', 'Year'])['Position'].shift(3)
    
    # LapTime lags
    df['LapTime_lag1'] = df.groupby(['Driver', 'Race', 'Year'])['LapTime (s)'].shift(1)
    df['LapTime_lag2'] = df.groupby(['Driver', 'Race', 'Year'])['LapTime (s)'].shift(2)
    df['LapTime_lag3'] = df.groupby(['Driver', 'Race', 'Year'])['LapTime (s)'].shift(3)
    
    # Rolling average pace (using LapTime_Delta as a proxy for pace change)
    # Rolling average of LapTime_Delta over 3 laps
    df['Rolling_avg_pace'] = df.groupby(['Driver', 'Race', 'Year'])['LapTime_Delta'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Rolling degradation (rolling average of Cumulative_Degradation)
    df['Rolling_degradation'] = df.groupby(['Driver', 'Race', 'Year'])['Cumulative_Degradation'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Fill NaN values (from shifts) with 0
    # Only fill numeric columns with 0, leave string/categorical columns unchanged
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Drop Driver and Race columns (they're used for grouping but not needed as features)
    df = df.drop(['Driver', 'Race'], axis=1)
    
    # Convert Compound column to categorical with specified order
    categories = ['SOFT', 'MEDIUM', 'HARD']
    dtype = pd.CategoricalDtype(categories=categories, ordered=True)
    df['Compound'] = df['Compound'].astype(dtype)
    
    # Impute missing values in Compound column
    if is_training:
        df['Compound'] = imputer.fit_transform(df['Compound'].values.reshape(-1, 1)).flatten()
    else:
        df['Compound'] = imputer.transform(df['Compound'].values.reshape(-1, 1)).flatten()
    
    # Encode Compound column with OrdinalEncoder
    if is_training:
        df['Compound_encoded'] = encoder.fit_transform(df['Compound'].to_numpy().reshape(-1, 1)).flatten()
    else:
        df['Compound_encoded'] = encoder.transform(df['Compound'].to_numpy().reshape(-1, 1)).flatten()
    
    # Drop the original Compound column
    df.drop('Compound', axis=1, inplace=True)
    
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
    Create lagged features for Position, LapTime, and degradation metrics.
    
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
    
    # LapTime lags
    df['LapTime_lag1'] = df.groupby(group_cols)['LapTime (s)'].shift(1)
    df['LapTime_lag2'] = df.groupby(group_cols)['LapTime (s)'].shift(2)
    df['LapTime_lag3'] = df.groupby(group_cols)['LapTime (s)'].shift(3)
    
    # Rolling average pace (using LapTime_Delta as a proxy for pace change)
    df['Rolling_avg_pace'] = df.groupby(group_cols)['LapTime_Delta'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Rolling degradation (rolling average of Cumulative_Degradation)
    df['Rolling_degradation'] = df.groupby(group_cols)['Cumulative_Degradation'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Fill NaN values (from shifts) with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df