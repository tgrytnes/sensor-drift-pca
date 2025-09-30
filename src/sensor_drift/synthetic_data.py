"""Generate synthetic sensor data for testing and development"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta


def generate_synthetic_sensor_data(
    n_samples: int = 1000,
    n_sensors: int = 128,
    n_chemicals: int = 6,
    n_batches: int = 5,
    drift_rate: float = 0.1,
    noise_level: float = 0.1,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """Generate synthetic gas sensor data with drift

    Args:
        n_samples: Total number of samples to generate
        n_sensors: Number of sensor features
        n_chemicals: Number of different chemical types
        n_batches: Number of temporal batches
        drift_rate: Rate of drift per batch (0.0 = no drift, 1.0 = extreme drift)
        noise_level: Amount of random noise to add
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic sensor data including drift
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create base patterns for each chemical (intrinsic signatures)
    # Each chemical has a characteristic pattern in low-dimensional space
    n_intrinsic_dims = 8  # True dimensionality
    chemical_patterns = np.random.randn(n_chemicals, n_intrinsic_dims) * 0.5

    # Create projection matrix from low-dim to high-dim sensor space using SVD
    # This ensures numerical stability
    U = np.random.randn(n_sensors, n_intrinsic_dims)
    U, _ = np.linalg.qr(U)  # Orthogonalize
    projection = U[:, :n_intrinsic_dims].T * 0.5  # Scale down to prevent overflow

    # Generate data for each batch
    data_list = []
    samples_per_batch = n_samples // n_batches

    for batch_idx in range(n_batches):
        # Create drift transformation for this batch
        drift_magnitude = batch_idx * drift_rate * 0.5  # Scale down drift

        # Translation drift (shift in sensor response)
        translation_drift = np.random.randn(n_sensors) * drift_magnitude * 0.2

        # Rotation drift (change in sensor sensitivity patterns)
        if drift_magnitude > 0:
            # Create small rotation using Givens rotations for stability
            angle = min(drift_magnitude * 0.1, 0.3)  # Cap maximum rotation
            rotation_matrix = np.eye(n_intrinsic_dims)

            # Apply rotation only to first two dimensions for simplicity
            c, s = np.cos(angle), np.sin(angle)
            rotation_matrix[0:2, 0:2] = np.array([[c, -s], [s, c]])
        else:
            rotation_matrix = np.eye(n_intrinsic_dims)

        # Generate samples for this batch
        batch_data = []
        batch_labels = []

        for _ in range(samples_per_batch):
            # Randomly select a chemical
            chemical_idx = np.random.randint(n_chemicals)

            # Get base pattern and apply rotation drift
            base_pattern = chemical_patterns[chemical_idx]
            drifted_pattern = rotation_matrix @ base_pattern

            # Add some variability to the pattern (within-class variation)
            pattern_variation = drifted_pattern + np.random.randn(n_intrinsic_dims) * 0.1

            # Project to high-dimensional sensor space
            sensor_response = pattern_variation @ projection

            # Apply translation drift
            sensor_response += translation_drift

            # Add measurement noise
            sensor_response += np.random.randn(n_sensors) * noise_level

            batch_data.append(sensor_response)
            batch_labels.append(chemical_idx)

        # Create DataFrame for this batch
        batch_df = pd.DataFrame(
            batch_data,
            columns=[f'sensor_{i:03d}' for i in range(n_sensors)]
        )
        batch_df['chemical'] = batch_labels
        batch_df['batch'] = batch_idx + 1  # Batch IDs start from 1

        # Add temporal information
        base_date = datetime(2020, 1, 1) + timedelta(days=batch_idx * 180)  # 6 months apart
        batch_df['timestamp'] = [
            base_date + timedelta(hours=i)
            for i in range(len(batch_df))
        ]

        data_list.append(batch_df)

    # Combine all batches
    df = pd.concat(data_list, ignore_index=True)

    # Map chemical indices to names
    chemical_names = ['Ammonia', 'Acetone', 'Ethylene', 'Ethanol', 'Toluene', 'Acetaldehyde']
    df['chemical_name'] = df['chemical'].map(
        {i: chemical_names[i % len(chemical_names)] for i in range(n_chemicals)}
    )

    return df


def create_batch_dict(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Convert DataFrame to dictionary of batch matrices

    Args:
        df: DataFrame with sensor data and batch column

    Returns:
        Dictionary mapping batch IDs to feature matrices
    """
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    batches = {}

    for batch_id in df['batch'].unique():
        batch_data = df[df['batch'] == batch_id][sensor_cols].values
        batches[batch_id] = batch_data

    return batches


def create_label_dict(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Convert DataFrame to dictionary of batch labels

    Args:
        df: DataFrame with chemical labels and batch column

    Returns:
        Dictionary mapping batch IDs to label arrays
    """
    labels = {}

    for batch_id in df['batch'].unique():
        batch_labels = df[df['batch'] == batch_id]['chemical'].values
        labels[batch_id] = batch_labels

    return labels


def verify_drift(df: pd.DataFrame) -> Dict:
    """Verify that drift exists in the synthetic data

    Args:
        df: DataFrame with synthetic sensor data

    Returns:
        Dictionary with drift statistics
    """
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

    # Calculate mean sensor responses per batch
    batch_means = df.groupby('batch')[sensor_cols].mean()

    # Calculate drift as Euclidean distance between consecutive batch means
    drift_distances = []
    for i in range(len(batch_means) - 1):
        dist = np.linalg.norm(batch_means.iloc[i] - batch_means.iloc[i + 1])
        drift_distances.append(dist)

    # Calculate drift per chemical
    chemical_drift = {}
    for chemical in df['chemical'].unique():
        chem_data = df[df['chemical'] == chemical]
        chem_batch_means = chem_data.groupby('batch')[sensor_cols].mean()

        if len(chem_batch_means) > 1:
            chem_drifts = []
            for i in range(len(chem_batch_means) - 1):
                dist = np.linalg.norm(
                    chem_batch_means.iloc[i] - chem_batch_means.iloc[i + 1]
                )
                chem_drifts.append(dist)
            chemical_drift[chemical] = np.mean(chem_drifts)

    return {
        'overall_drift': drift_distances,
        'mean_drift': np.mean(drift_distances) if drift_distances else 0,
        'chemical_drift': chemical_drift,
        'batch_means_std': batch_means.std().mean()
    }


if __name__ == "__main__":
    # Generate sample data
    print("Generating synthetic sensor data...")
    df = generate_synthetic_sensor_data(
        n_samples=2000,
        n_sensors=128,
        n_chemicals=6,
        n_batches=5,
        drift_rate=0.15,
        noise_level=0.1
    )

    print(f"\nDataset shape: {df.shape}")
    print(f"Batches: {sorted(df['batch'].unique())}")
    print(f"Chemicals: {sorted(df['chemical_name'].unique())}")

    # Verify drift
    drift_stats = verify_drift(df)
    print(f"\nMean drift between batches: {drift_stats['mean_drift']:.3f}")
    print(f"Batch means std: {drift_stats['batch_means_std']:.3f}")

    # Save sample data - use absolute path from project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent  # Go up to project root
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'synthetic_sensor_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSynthetic data saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")