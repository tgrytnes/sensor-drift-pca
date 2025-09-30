"""Utility functions to load sensor data"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_synthetic_data(file_name: str = 'synthetic_sensor_data.csv') -> pd.DataFrame:
    """Load synthetic sensor data from data/processed directory

    Args:
        file_name: Name of the CSV file to load

    Returns:
        DataFrame with sensor data
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'processed' / file_name

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            f"Please run 'python3 src/sensor_drift/synthetic_data.py' to generate it."
        )

    # Load data with proper date parsing
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

    print(f"Loaded data shape: {df.shape}")
    print(f"Batches: {sorted(df['batch'].unique())}")
    print(f"Chemicals: {sorted(df['chemical_name'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def get_sensor_columns(df: pd.DataFrame) -> list:
    """Extract sensor column names from dataframe

    Args:
        df: DataFrame with sensor data

    Returns:
        List of sensor column names
    """
    return [col for col in df.columns if col.startswith('sensor_')]


def split_by_batch(df: pd.DataFrame) -> dict:
    """Split dataframe into dictionary by batch

    Args:
        df: DataFrame with 'batch' column

    Returns:
        Dictionary mapping batch ID to dataframe
    """
    batches = {}
    for batch_id in df['batch'].unique():
        batches[batch_id] = df[df['batch'] == batch_id].copy()
    return batches


if __name__ == "__main__":
    # Test loading the data
    print("Testing data loading...")
    df = load_synthetic_data()
    print(f"\nFirst few rows:")
    print(df.head())

    sensor_cols = get_sensor_columns(df)
    print(f"\nNumber of sensor columns: {len(sensor_cols)}")
    print(f"First 5 sensors: {sensor_cols[:5]}")

    batches = split_by_batch(df)
    print(f"\nBatch sizes:")
    for batch_id, batch_df in batches.items():
        print(f"  Batch {batch_id}: {len(batch_df)} samples")