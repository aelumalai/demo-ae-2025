"""
Data preprocessing utilities for Module 5 Assignment.

This module contains helper functions for data loading,
cleaning, and preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        df_copy = df_copy.dropna()
    else:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_copy[col].isnull().sum() > 0:
                if strategy == 'mean':
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
    
    return df_copy


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (np.array or pd.DataFrame): Training features
        X_test (np.array or pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def encode_categorical(df, columns):
    """
    Encode categorical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to encode
        
    Returns:
        tuple: (encoded_df, encoders_dict)
    """
    df_copy = df.copy()
    encoders = {}
    
    for col in columns:
        if col in df_copy.columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
    
    return df_copy, encoders
