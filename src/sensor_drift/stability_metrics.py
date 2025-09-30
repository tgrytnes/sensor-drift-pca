"""Stability metrics and clustering evaluation for sensor drift analysis"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import warnings


def evaluate_clustering_stability(X_list: List[np.ndarray],
                                 labels_list: List[np.ndarray],
                                 metric: str = 'silhouette') -> List[float]:
    """Evaluate clustering quality across multiple batches

    Args:
        X_list: List of feature matrices (one per batch)
        labels_list: List of cluster labels
        metric: 'silhouette' or 'davies_bouldin'

    Returns:
        List of metric scores for each batch
    """
    scores = []

    for X, labels in zip(X_list, labels_list):
        if len(np.unique(labels)) < 2:
            warnings.warn("Less than 2 clusters found, skipping batch")
            scores.append(np.nan)
            continue

        if metric == 'silhouette':
            score = silhouette_score(X, labels)
        elif metric == 'davies_bouldin':
            score = davies_bouldin_score(X, labels)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    return scores


def compare_subspace_clustering(X: np.ndarray,
                               labels: np.ndarray,
                               subspace_dims: List[List[int]]) -> Dict:
    """Compare clustering quality in different subspaces

    Args:
        X: Full feature matrix
        labels: True cluster labels
        subspace_dims: List of dimension indices for each subspace

    Returns:
        Dictionary with clustering metrics for each subspace
    """
    results = {}

    for i, dims in enumerate(subspace_dims):
        X_sub = X[:, dims]

        # Evaluate clustering with true labels
        if len(np.unique(labels)) >= 2:
            sil_score = silhouette_score(X_sub, labels)
            db_score = davies_bouldin_score(X_sub, labels)
        else:
            sil_score = np.nan
            db_score = np.nan

        results[f'subspace_{i}'] = {
            'dimensions': dims,
            'n_dims': len(dims),
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score
        }

    return results


def cluster_purity(labels_true: np.ndarray,
                  labels_pred: np.ndarray) -> float:
    """Calculate cluster purity metric

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels

    Returns:
        Purity score between 0 and 1
    """
    contingency_matrix = np.zeros((len(np.unique(labels_pred)),
                                  len(np.unique(labels_true))))

    for i, label_pred in enumerate(np.unique(labels_pred)):
        for j, label_true in enumerate(np.unique(labels_true)):
            contingency_matrix[i, j] = np.sum((labels_pred == label_pred) &
                                             (labels_true == label_true))

    purity = np.sum(np.max(contingency_matrix, axis=1)) / len(labels_true)
    return purity


def compute_drift_distance(centroid1: np.ndarray,
                          centroid2: np.ndarray,
                          metric: str = 'euclidean') -> float:
    """Compute distance between cluster centroids

    Args:
        centroid1, centroid2: Cluster centroids
        metric: Distance metric ('euclidean' or 'mahalanobis')

    Returns:
        Distance between centroids
    """
    if metric == 'euclidean':
        return np.linalg.norm(centroid1 - centroid2)
    elif metric == 'cosine':
        cos_sim = np.dot(centroid1, centroid2) / (
            np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
        return 1 - cos_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


def measure_batch_alignment(batch_data: Dict[str, np.ndarray],
                           labels: Dict[str, np.ndarray]) -> np.ndarray:
    """Measure alignment between batches using centroid distances

    Args:
        batch_data: Dictionary mapping batch IDs to feature matrices
        labels: Dictionary mapping batch IDs to label arrays

    Returns:
        Matrix of pairwise batch distances
    """
    batch_ids = list(batch_data.keys())
    n_batches = len(batch_ids)
    distance_matrix = np.zeros((n_batches, n_batches))

    for i, batch_i in enumerate(batch_ids):
        for j, batch_j in enumerate(batch_ids):
            if i >= j:
                continue

            X_i = batch_data[batch_i]
            X_j = batch_data[batch_j]
            labels_i = labels[batch_i]
            labels_j = labels[batch_j]

            # Find common chemical labels
            common_labels = set(labels_i) & set(labels_j)

            distances = []
            for label in common_labels:
                # Compute centroids for this chemical in both batches
                centroid_i = X_i[labels_i == label].mean(axis=0)
                centroid_j = X_j[labels_j == label].mean(axis=0)

                # Compute distance
                dist = compute_drift_distance(centroid_i, centroid_j)
                distances.append(dist)

            if distances:
                # Average distance across all common chemicals
                avg_distance = np.mean(distances)
                distance_matrix[i, j] = avg_distance
                distance_matrix[j, i] = avg_distance

    return distance_matrix


def temporal_consistency_score(predictions_over_time: List[np.ndarray]) -> float:
    """Measure consistency of predictions across time batches

    Args:
        predictions_over_time: List of prediction arrays for each batch

    Returns:
        Consistency score between 0 and 1
    """
    if len(predictions_over_time) < 2:
        return 1.0

    consistencies = []
    for i in range(len(predictions_over_time) - 1):
        pred1 = predictions_over_time[i]
        pred2 = predictions_over_time[i + 1]

        # Assuming same ordering of samples or matched samples
        min_len = min(len(pred1), len(pred2))
        agreement = np.mean(pred1[:min_len] == pred2[:min_len])
        consistencies.append(agreement)

    return np.mean(consistencies)


def stability_index(angles: np.ndarray,
                   threshold: float = 15.0) -> float:
    """Compute overall stability index from PC angles

    Args:
        angles: Matrix of angles between PCs across batches
        threshold: Angle threshold for "stable" component

    Returns:
        Stability index between 0 (unstable) and 1 (perfectly stable)
    """
    # Proportion of angles below threshold
    stable_proportion = np.mean(angles < threshold)

    # Penalize based on mean angle
    mean_angle = np.mean(angles)
    angle_penalty = 1 - (mean_angle / 90)  # Normalize to [0, 1]

    # Combined stability index
    stability = stable_proportion * angle_penalty

    return stability