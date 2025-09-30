"""Data preprocessing utilities for sensor drift analysis"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path


def load_sensor_data(file_path: str) -> pd.DataFrame:
    """Load gas sensor array drift dataset

    Args:
        file_path: Path to the dataset file

    Returns:
        DataFrame with sensor readings and metadata
    """
    # Implementation will depend on actual data format
    # This is a placeholder for the actual loading logic
    pass


def extract_batches(df: pd.DataFrame, batch_col: str = 'batch') -> dict:
    """Separate data into temporal batches

    Args:
        df: Full dataset
        batch_col: Column name containing batch identifiers

    Returns:
        Dictionary mapping batch IDs to DataFrames
    """
    batches = {}
    for batch_id in df[batch_col].unique():
        batches[batch_id] = df[df[batch_col] == batch_id].copy()
    return batches


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """Normalize sensor features

    Args:
        X: Feature matrix (n_samples, n_features)
        method: 'standard' for z-score, 'minmax' for min-max scaling

    Returns:
        Normalized features and scaling parameters
    """
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, params


def prepare_sensor_matrix(df: pd.DataFrame,
                         sensor_cols: Optional[List[str]] = None,
                         target_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to feature matrix and labels

    Args:
        df: Sensor data DataFrame
        sensor_cols: List of column names for sensor readings
        target_col: Column name for target labels (chemical type)

    Returns:
        Feature matrix X and label vector y
    """
    if sensor_cols is None:
        # Assume all numeric columns except target are sensors
        sensor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in sensor_cols:
            sensor_cols.remove(target_col)

    X = df[sensor_cols].values

    if target_col:
        y = df[target_col].values
    else:
        y = None

    return X, y


def handle_missing_values(X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """Handle missing values in sensor data

    Args:
        X: Feature matrix with potential NaN values
        strategy: 'mean', 'median', 'forward_fill', or 'drop'

    Returns:
        Cleaned feature matrix
    """
    if strategy == 'mean':
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
    elif strategy == 'median':
        col_medians = np.nanmedian(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_medians, nan_indices[1])
    elif strategy == 'drop':
        # Remove rows with any NaN
        X = X[~np.isnan(X).any(axis=1)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X