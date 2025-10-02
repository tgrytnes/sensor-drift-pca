"""PCA analysis and stability metrics for sensor drift"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data_preprocessing import normalize_features


def compute_pca(X: np.ndarray, n_components: Optional[int] = None) -> Tuple[PCA, np.ndarray]:
    """Compute PCA on sensor data

    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep (None for all)

    Returns:
        Fitted PCA object and transformed data
    """
    pca = PCA(n_components=n_components, svd_solver='full')
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
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            X_transformed = pca_reference.transform(X_batch)

        if not np.isfinite(X_transformed).all():
            raise ValueError(f"Non-finite values encountered when projecting batch {batch_id}")

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


def infer_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Infer sensor feature columns from processed dataset"""
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    if sensor_cols:
        return sensor_cols

    sensor_pattern = re.compile(r'^S\d{2}_F\d{1,2}_[A-Za-z0-9_]+$')
    sensor_cols = [col for col in df.columns if sensor_pattern.match(col)]
    if sensor_cols:
        return sensor_cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    reserved = {'batch', 'gas_type', 'target', 'label', 'concentration'}
    sensor_cols = [col for col in numeric_cols if col not in reserved]

    if not sensor_cols:
        raise ValueError("Unable to infer sensor feature columns from dataset")

    return sensor_cols


def run_sensor_pca_analysis(
    df: pd.DataFrame,
    sensor_cols: List[str],
    batch_col: str = 'batch',
    label_col: Optional[str] = 'gas_name',
    n_components: int = 50,
    stability_components: int = 10,
    stability_threshold: float = 15.0,
    batch_mean_threshold: float = 0.5
) -> Dict:
    """Execute PCA workflow on processed sensor dataset"""

    if n_components <= 0:
        raise ValueError("n_components must be positive")

    if stability_components <= 0:
        raise ValueError("stability_components must be positive")

    X = df[sensor_cols].to_numpy(dtype=float)
    X_norm, norm_params = normalize_features(X, method='standard')

    max_components = min(n_components, X_norm.shape[0], X_norm.shape[1])
    pca_global, X_pca = compute_pca(X_norm, n_components=max_components)
    eigen_analysis = analyze_eigenvalue_spectrum(pca_global)

    components_for_variance = {
        key: int(val) for key, val in eigen_analysis['components_for_variance'].items()
    }

    batch_ids = sorted(df[batch_col].unique())
    batch_sizes = {}
    batch_arrays: Dict = {}
    batch_labels: Dict = {}
    pca_batches: List[PCA] = []
    batch_mean_scores = []

    for batch_id in batch_ids:
        mask = df[batch_col] == batch_id
        batch_sizes[str(batch_id)] = int(mask.sum())

        X_batch = X_norm[mask]
        batch_arrays[batch_id] = X_batch

        if X_batch.size == 0:
            continue

        batch_mean_scores.append(X_pca[mask])

        n_batch_components = min(stability_components, X_batch.shape[0], X_batch.shape[1])
        if n_batch_components < 1:
            continue

        pca_batch, _ = compute_pca(X_batch, n_components=n_batch_components)
        pca_batches.append(pca_batch)

        if label_col and label_col in df.columns:
            batch_labels[batch_id] = df.loc[mask, label_col].to_numpy()

    if batch_mean_scores:
        batch_means = np.vstack([scores.mean(axis=0) for scores in batch_mean_scores])
        component_batch_std = batch_means.std(axis=0)
        stable_components_by_mean = np.where(component_batch_std < batch_mean_threshold)[0]
    else:
        component_batch_std = np.array([])
        stable_components_by_mean = np.array([])

    if len(pca_batches) >= 2:
        stability_angles = measure_pc_stability(pca_batches, n_components=stability_components)
        mean_angles = stability_angles.mean(axis=0) if stability_angles.size else np.array([])
        stable_components = find_stable_subspace(stability_angles, stability_threshold) if stability_angles.size else []
    else:
        stability_angles = np.zeros((0, stability_components))
        mean_angles = np.array([])
        stable_components = []

    if pca_batches:
        chemical_labels = batch_labels if batch_labels else None
        drift_df = track_drift_in_pc_space(batch_arrays, pca_batches[0], chemical_labels)
    else:
        drift_df = pd.DataFrame()

    drift_velocities = {}
    overall_velocity = None
    if not drift_df.empty and 'drift_velocity' in drift_df.columns:
        velocity_series = drift_df['drift_velocity'].dropna()
        if not velocity_series.empty:
            overall_velocity = float(velocity_series.mean())

        for chem, values in drift_df.groupby('chemical')['drift_velocity']:
            mean_val = values.dropna().mean()
            if not np.isnan(mean_val):
                drift_velocities[str(chem)] = float(mean_val)

    stability_df = pd.DataFrame()
    if stability_angles.size:
        n_cols = stability_angles.shape[1]
        stability_df = pd.DataFrame(
            stability_angles,
            columns=[f'PC{i+1}' for i in range(n_cols)]
        )

    summary = {
        'dataset': {
            'n_samples': int(len(df)),
            'n_features': int(len(sensor_cols)),
            'batch_column': batch_col,
            'label_column': label_col if label_col and label_col in df.columns else None,
            'batch_sample_counts': batch_sizes,
        },
        'normalization': {
            'mean': norm_params['mean'].tolist(),
            'std': norm_params['std'].tolist()
        },
        'global_pca': {
            'n_components_fitted': int(pca_global.n_components_),
            'explained_variance_ratio_first10': eigen_analysis['explained_variance_ratio'][:10].tolist(),
            'cumulative_variance_first10': eigen_analysis['cumulative_variance'][:10].tolist(),
            'kaiser_n_components': int(eigen_analysis['kaiser_n_components']),
            'elbow_index': int(eigen_analysis['elbow_index']),
            'components_for_variance': components_for_variance
        },
        'stability': {
            'mean_angles_degrees': mean_angles.tolist(),
            'stable_components_zero_indexed': [int(idx) for idx in stable_components],
            'stable_components_one_indexed': [int(idx) + 1 for idx in stable_components],
            'stability_threshold_degrees': float(stability_threshold)
        },
        'component_stability': {
            'batch_mean_std': component_batch_std.tolist(),
            'stable_components_zero_indexed': [int(idx) for idx in stable_components_by_mean],
            'stable_components_one_indexed': [int(idx) + 1 for idx in stable_components_by_mean],
            'batch_mean_threshold': float(batch_mean_threshold)
        },
        'drift': {
            'overall_average_velocity': overall_velocity,
            'average_velocity_by_chemical': drift_velocities
        }
    }

    return {
        'summary': summary,
        'stability_angles': stability_angles,
        'stability_df': stability_df,
        'drift_df': drift_df,
        'global_scores': X_pca
    }


def _pretty_print_summary(summary: Dict) -> None:
    """Print key metrics to stdout"""
    dataset = summary['dataset']
    global_pca_summary = summary['global_pca']
    stability = summary['stability']
    drift = summary['drift']

    print("=" * 70)
    print("DATASET")
    print(f"Rows: {dataset['n_samples']} | Features: {dataset['n_features']} | Batches: {len(dataset['batch_sample_counts'])}")
    print(f"Batch counts: {dataset['batch_sample_counts']}")

    print("\nGLOBAL PCA")
    print(f"Components fitted: {global_pca_summary['n_components_fitted']}")
    print(f"Variance for 90/95/99%: {global_pca_summary['components_for_variance']}")
    print(f"Elbow index: {global_pca_summary['elbow_index']} | Kaiser: {global_pca_summary['kaiser_n_components']}")

    print("\nSTABILITY")
    if stability['mean_angles_degrees']:
        top_angles = stability['mean_angles_degrees'][:5]
        print(f"Mean angles (first 5 PCs): {[round(val, 2) for val in top_angles]}")
        print(f"Stable PCs (<{stability['stability_threshold_degrees']}°): {stability['stable_components_one_indexed']}")
    else:
        print("Not enough batches to compute stability metrics")

    component_stability = summary['component_stability']
    if component_stability['batch_mean_std']:
        stable_mean = component_stability['stable_components_one_indexed']
        print("\nBATCH-MEAN STABILITY")
        print(f"Std threshold: {component_stability['batch_mean_threshold']}")
        print(f"Stable PCs by batch means: {stable_mean}")
        print(f"First 5 stds: {[round(val, 3) for val in component_stability['batch_mean_std'][:5]]}")

    print("\nDRIFT")
    if drift['overall_average_velocity'] is not None:
        print(f"Overall average drift velocity: {drift['overall_average_velocity']:.4f}")
        print(f"Velocity by chemical: {drift['average_velocity_by_chemical']}")
    else:
        print("Drift metrics unavailable (insufficient data)")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments"""
    project_root = Path(__file__).resolve().parents[2]
    default_data = project_root / 'data' / 'processed' / 'sensor_data.csv'
    default_output = project_root / 'results' / 'pca_analysis'

    parser = argparse.ArgumentParser(description="Run PCA analysis on sensor drift dataset")
    parser.add_argument('--data', type=Path, default=default_data, help='Path to processed sensor dataset CSV')
    parser.add_argument('--batch-col', default='batch', help='Name of batch column in dataset')
    parser.add_argument('--label-col', default='gas_name', help='Name of label column (optional)')
    parser.add_argument('--components', type=int, default=50, help='Maximum number of PCA components to fit')
    parser.add_argument('--stability-components', type=int, default=10, help='Number of components to evaluate for stability')
    parser.add_argument('--stability-threshold', type=float, default=15.0, help='Stability angle threshold in degrees')
    parser.add_argument('--batch-mean-threshold', type=float, default=0.5, help='Std threshold for identifying stable PCs via batch means')
    parser.add_argument('--output-dir', type=Path, default=default_output, help='Directory to store analysis artifacts')
    parser.add_argument('--no-save', action='store_true', help='Skip writing artifacts to disk')

    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution"""
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data}")

    df = pd.read_csv(args.data)
    sensor_cols = infer_sensor_columns(df)

    results = run_sensor_pca_analysis(
        df=df,
        sensor_cols=sensor_cols,
        batch_col=args.batch_col,
        label_col=args.label_col if args.label_col in df.columns else None,
        n_components=args.components,
        stability_components=args.stability_components,
        stability_threshold=args.stability_threshold,
        batch_mean_threshold=args.batch_mean_threshold
    )

    summary = results['summary']
    _pretty_print_summary(summary)

    if not args.no_save:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / 'summary.json'
        summary_path.write_text(json.dumps(summary, indent=2))

        stability_df: pd.DataFrame = results['stability_df']
        if not stability_df.empty:
            stability_df.to_csv(output_dir / 'pc_stability.csv', index=False)

        drift_df: pd.DataFrame = results['drift_df']
        if not drift_df.empty:
            drift_df.to_csv(output_dir / 'drift_metrics.csv', index=False)

        scores = results['global_scores']
        np.save(output_dir / 'global_pca_scores.npy', scores)


if __name__ == '__main__':
    main()
