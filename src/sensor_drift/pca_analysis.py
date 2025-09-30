"""PCA analysis and stability metrics for sensor drift"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional
import pandas as pd


def compute_pca(X: np.ndarray, n_components: Optional[int] = None) -> Tuple[PCA, np.ndarray]:
    """Compute PCA on sensor data

    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep (None for all)

    Returns:
        Fitted PCA object and transformed data
    """
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    return pca, X_transformed


def analyze_eigenvalue_spectrum(pca: PCA) -> Dict:
    """Analyze eigenvalue spectrum for dimensionality selection

    Args:
        pca: Fitted PCA object

    Returns:
        Dictionary with eigenvalue analysis metrics
    """
    eigenvalues = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    # Find elbow point using second derivative
    diffs = np.diff(eigenvalues)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 2  # +2 to account for double diff

    # Kaiser criterion (eigenvalues > 1 for standardized data)
    kaiser_n = np.sum(eigenvalues > 1)

    # Find components needed for 90%, 95%, 99% variance
    var_thresholds = [0.90, 0.95, 0.99]
    components_for_var = {}
    for threshold in var_thresholds:
        n_comp = np.argmax(cumulative_var >= threshold) + 1
        components_for_var[f'{int(threshold*100)}%'] = int(n_comp)

    return {
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_var_ratio,
        'cumulative_variance': cumulative_var,
        'elbow_index': int(elbow_idx),
        'kaiser_n_components': int(kaiser_n),
        'components_for_variance': components_for_var
    }


def measure_pc_stability(pca_list: List[PCA],
                         n_components: int = 10) -> np.ndarray:
    """Measure angular stability of principal components across batches

    Args:
        pca_list: List of fitted PCA objects (one per batch)
        n_components: Number of components to analyze

    Returns:
        Matrix of angular distances between PCs across batches
    """
    n_batches = len(pca_list)
    n_comp_min = min(n_components, min(pca.n_components_ for pca in pca_list))

    # Store angles between consecutive batches
    angles = np.zeros((n_batches - 1, n_comp_min))

    for i in range(n_batches - 1):
        pc1 = pca_list[i].components_[:n_comp_min]
        pc2 = pca_list[i + 1].components_[:n_comp_min]

        for j in range(n_comp_min):
            # Compute angle between j-th principal component
            cos_angle = np.abs(np.dot(pc1[j], pc2[j]))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            angle = np.degrees(np.arccos(cos_angle))
            angles[i, j] = angle

    return angles


def compute_subspace_angles(pca1: PCA, pca2: PCA,
                           n_components: int = 5) -> np.ndarray:
    """Compute principal angles between two PCA subspaces

    Args:
        pca1, pca2: Two fitted PCA objects
        n_components: Number of dimensions for subspace

    Returns:
        Principal angles between subspaces (in degrees)
    """
    # Get the first n_components principal components
    U1 = pca1.components_[:n_components].T  # (n_features, n_components)
    U2 = pca2.components_[:n_components].T

    # Compute SVD of U1.T @ U2
    _, s, _ = np.linalg.svd(U1.T @ U2)

    # Principal angles are arccos of singular values
    s = np.clip(s, -1, 1)  # Handle numerical errors
    angles = np.degrees(np.arccos(s))

    return angles


def track_drift_in_pc_space(data_batches: Dict[str, np.ndarray],
                           pca_reference: PCA,
                           chemical_labels: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Track how clusters drift in PC space over time

    Args:
        data_batches: Dictionary mapping batch IDs to feature matrices
        pca_reference: Reference PCA (typically from first batch)
        chemical_labels: Optional dictionary of labels for each batch

    Returns:
        DataFrame with drift metrics for each batch and chemical
    """
    drift_results = []

    for batch_id, X_batch in data_batches.items():
        # Project batch data onto reference PC space
        X_transformed = pca_reference.transform(X_batch)

        if chemical_labels and batch_id in chemical_labels:
            labels = chemical_labels[batch_id]
            unique_labels = np.unique(labels)

            for label in unique_labels:
                mask = labels == label
                cluster_data = X_transformed[mask]

                # Compute cluster centroid
                centroid = cluster_data.mean(axis=0)

                # Compute cluster spread (covariance trace)
                if len(cluster_data) > 1:
                    spread = np.trace(np.cov(cluster_data.T))
                else:
                    spread = 0

                drift_results.append({
                    'batch': batch_id,
                    'chemical': label,
                    'centroid': centroid,
                    'spread': spread,
                    'n_samples': len(cluster_data)
                })
        else:
            # Overall batch statistics
            centroid = X_transformed.mean(axis=0)
            spread = np.trace(np.cov(X_transformed.T))

            drift_results.append({
                'batch': batch_id,
                'chemical': 'all',
                'centroid': centroid,
                'spread': spread,
                'n_samples': len(X_transformed)
            })

    df_drift = pd.DataFrame(drift_results)

    # Compute drift velocities if we have temporal ordering
    if len(drift_results) > 1:
        # Sort by batch (assuming numeric or chronological ordering)
        df_drift = df_drift.sort_values('batch')

        # Compute centroid displacement between consecutive batches
        for chem in df_drift['chemical'].unique():
            chem_data = df_drift[df_drift['chemical'] == chem].copy()
            if len(chem_data) > 1:
                centroids = np.vstack(chem_data['centroid'].values)
                displacements = np.diff(centroids, axis=0)
                velocities = np.linalg.norm(displacements, axis=1)
                df_drift.loc[df_drift['chemical'] == chem, 'drift_velocity'] = \
                    [0] + velocities.tolist()

    return df_drift


def find_stable_subspace(stability_angles: np.ndarray,
                        stability_threshold: float = 15.0) -> List[int]:
    """Identify stable principal components based on angular stability

    Args:
        stability_angles: Matrix of angles between PCs across batches
        stability_threshold: Maximum angle (degrees) for component to be "stable"

    Returns:
        List of stable component indices
    """
    # Average angle across all batch transitions
    mean_angles = stability_angles.mean(axis=0)

    # Find components with mean angle below threshold
    stable_components = np.where(mean_angles < stability_threshold)[0]

    return stable_components.tolist()


def parallel_analysis(X: np.ndarray, n_iterations: int = 100) -> int:
    """Parallel analysis for determining significant components

    Args:
        X: Feature matrix
        n_iterations: Number of random permutations

    Returns:
        Number of significant components
    """
    n_samples, n_features = X.shape

    # Compute eigenvalues from actual data
    pca_actual = PCA()
    pca_actual.fit(X)
    actual_eigenvalues = pca_actual.explained_variance_

    # Generate random data and compute eigenvalues
    random_eigenvalues = np.zeros((n_iterations, n_features))
    for i in range(n_iterations):
        X_random = np.random.randn(n_samples, n_features)
        pca_random = PCA()
        pca_random.fit(X_random)
        random_eigenvalues[i] = pca_random.explained_variance_

    # 95th percentile of random eigenvalues
    random_95 = np.percentile(random_eigenvalues, 95, axis=0)

    # Number of components where actual > random 95th percentile
    n_significant = np.sum(actual_eigenvalues > random_95)

    return int(n_significant)