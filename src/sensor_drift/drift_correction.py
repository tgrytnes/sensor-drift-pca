"""Drift correction methods including Procrustes alignment"""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, List


def procrustes_alignment(X_source: np.ndarray,
                         X_target: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Align source data to target using Procrustes analysis

    Args:
        X_source: Source data to be aligned (n_samples, n_features)
        X_target: Target/reference data (n_samples, n_features)

    Returns:
        Aligned source data and transformation parameters
    """
    # Center both datasets
    mean_source = X_source.mean(axis=0)
    mean_target = X_target.mean(axis=0)

    X_source_centered = X_source - mean_source
    X_target_centered = X_target - mean_target

    # Find optimal rotation matrix
    R, scale = orthogonal_procrustes(X_source_centered, X_target_centered)

    # Apply transformation
    X_aligned = X_source_centered @ R + mean_target

    transformation = {
        'rotation': R,
        'scale': scale,
        'source_mean': mean_source,
        'target_mean': mean_target
    }

    return X_aligned, transformation


def align_pca_spaces(pca_source: PCA,
                    pca_target: PCA,
                    n_components: Optional[int] = None) -> np.ndarray:
    """Align PCA spaces using Procrustes on the loading matrices

    Args:
        pca_source: Source PCA model
        pca_target: Target/reference PCA model
        n_components: Number of components to align (default: all)

    Returns:
        Rotation matrix for aligning source to target PC space
    """
    if n_components is None:
        n_components = min(pca_source.n_components_,
                          pca_target.n_components_)

    # Get loading matrices (components)
    V_source = pca_source.components_[:n_components].T
    V_target = pca_target.components_[:n_components].T

    # Find optimal rotation
    R, _ = orthogonal_procrustes(V_source, V_target)

    return R


def correct_batch_drift(X_batch: np.ndarray,
                       X_reference: np.ndarray,
                       method: str = 'procrustes',
                       **kwargs) -> np.ndarray:
    """Correct drift in batch data using reference batch

    Args:
        X_batch: Data from drifted batch
        X_reference: Reference batch data
        method: Correction method ('procrustes', 'mean_shift', 'standardize')
        **kwargs: Additional method-specific parameters

    Returns:
        Corrected batch data
    """
    if method == 'procrustes':
        X_corrected, _ = procrustes_alignment(X_batch, X_reference)

    elif method == 'mean_shift':
        # Simple mean centering to reference
        mean_batch = X_batch.mean(axis=0)
        mean_ref = X_reference.mean(axis=0)
        X_corrected = X_batch - mean_batch + mean_ref

    elif method == 'standardize':
        # Match mean and standard deviation
        mean_batch = X_batch.mean(axis=0)
        std_batch = X_batch.std(axis=0)
        mean_ref = X_reference.mean(axis=0)
        std_ref = X_reference.std(axis=0)

        X_corrected = (X_batch - mean_batch) / (std_batch + 1e-8)
        X_corrected = X_corrected * std_ref + mean_ref

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return X_corrected


def incremental_drift_correction(batch_sequence: List[np.ndarray],
                                 reference_idx: int = 0) -> List[np.ndarray]:
    """Correct drift across a sequence of batches incrementally

    Args:
        batch_sequence: List of batch data matrices
        reference_idx: Index of reference batch (default: 0)

    Returns:
        List of corrected batch data
    """
    corrected = []
    reference = batch_sequence[reference_idx]

    for i, batch in enumerate(batch_sequence):
        if i == reference_idx:
            corrected.append(batch)
        else:
            # Correct to reference
            X_corrected = correct_batch_drift(batch, reference, method='procrustes')
            corrected.append(X_corrected)

    return corrected


def adaptive_drift_model(X_batches: Dict[str, np.ndarray],
                        timestamps: Optional[Dict[str, float]] = None) -> Dict:
    """Build adaptive model of drift over time

    Args:
        X_batches: Dictionary mapping batch IDs to data
        timestamps: Optional timestamps for each batch

    Returns:
        Dictionary containing drift model parameters
    """
    batch_ids = list(X_batches.keys())
    n_batches = len(batch_ids)

    # Use first batch as reference
    reference_batch = X_batches[batch_ids[0]]
    reference_mean = reference_batch.mean(axis=0)

    drift_vectors = []
    drift_magnitudes = []

    for i in range(1, n_batches):
        batch = X_batches[batch_ids[i]]
        batch_mean = batch.mean(axis=0)

        # Compute drift vector
        drift_vec = batch_mean - reference_mean
        drift_vectors.append(drift_vec)
        drift_magnitudes.append(np.linalg.norm(drift_vec))

    # Fit linear drift model if timestamps available
    drift_model = {
        'reference_batch': batch_ids[0],
        'drift_vectors': np.array(drift_vectors),
        'drift_magnitudes': np.array(drift_magnitudes)
    }

    if timestamps:
        # Fit linear regression for drift over time
        from sklearn.linear_model import LinearRegression
        times = np.array([timestamps[bid] for bid in batch_ids[1:]])
        times = times.reshape(-1, 1)

        # Fit drift magnitude over time
        lr = LinearRegression()
        lr.fit(times, drift_magnitudes)

        drift_model['temporal_model'] = {
            'slope': lr.coef_[0],
            'intercept': lr.intercept_,
            'r2_score': lr.score(times, drift_magnitudes)
        }

    return drift_model


def apply_drift_model(X_new: np.ndarray,
                     drift_model: Dict,
                     timestamp: Optional[float] = None) -> np.ndarray:
    """Apply learned drift model to correct new data

    Args:
        X_new: New data to correct
        drift_model: Drift model from adaptive_drift_model
        timestamp: Optional timestamp for temporal correction

    Returns:
        Corrected data
    """
    if timestamp and 'temporal_model' in drift_model:
        # Predict drift magnitude at this timestamp
        slope = drift_model['temporal_model']['slope']
        intercept = drift_model['temporal_model']['intercept']
        predicted_magnitude = slope * timestamp + intercept

        # Use average drift direction
        avg_drift_direction = drift_model['drift_vectors'].mean(axis=0)
        avg_drift_direction /= np.linalg.norm(avg_drift_direction)

        # Apply correction
        correction = predicted_magnitude * avg_drift_direction
        X_corrected = X_new - correction

    else:
        # Use average drift vector
        avg_drift = drift_model['drift_vectors'].mean(axis=0)
        X_corrected = X_new - avg_drift

    return X_corrected