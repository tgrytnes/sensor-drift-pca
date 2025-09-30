#!/usr/bin/env python3
"""Smoke test for sensor drift PCA analysis pipeline"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Since we're now in the same directory, we can use relative imports
from synthetic_data import (
    generate_synthetic_sensor_data,
    create_batch_dict,
    create_label_dict,
    verify_drift
)
from data_preprocessing import (
    normalize_features,
    prepare_sensor_matrix,
    handle_missing_values
)
from pca_analysis import (
    compute_pca,
    analyze_eigenvalue_spectrum,
    measure_pc_stability,
    track_drift_in_pc_space,
    find_stable_subspace
)
from stability_metrics import (
    evaluate_clustering_stability,
    compare_subspace_clustering,
    measure_batch_alignment
)
from drift_correction import (
    procrustes_alignment,
    correct_batch_drift,
    incremental_drift_correction
)


def main():
    """Run complete smoke test of the PCA drift analysis pipeline"""

    print("=" * 70)
    print("SENSOR DRIFT PCA ANALYSIS - SMOKE TEST")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n1. GENERATING SYNTHETIC SENSOR DATA")
    print("-" * 40)

    df = generate_synthetic_sensor_data(
        n_samples=2000,
        n_sensors=128,
        n_chemicals=6,
        n_batches=5,
        drift_rate=0.2,  # Significant drift
        noise_level=0.1,
        random_seed=42
    )

    print(f"✓ Generated dataset with shape: {df.shape}")
    print(f"  - Batches: {sorted(df['batch'].unique())}")
    print(f"  - Chemicals: {len(df['chemical'].unique())} types")
    print(f"  - Sensors: {len([c for c in df.columns if c.startswith('sensor_')])} features")

    # Verify drift exists
    drift_stats = verify_drift(df)
    print(f"✓ Verified drift: mean = {drift_stats['mean_drift']:.3f}")

    # Step 2: Data Preprocessing
    print("\n2. DATA PREPROCESSING")
    print("-" * 40)

    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    X, y = prepare_sensor_matrix(df, sensor_cols, 'chemical')
    print(f"✓ Prepared feature matrix: {X.shape}")

    # Normalize features
    X_norm, norm_params = normalize_features(X, method='standard')
    print(f"✓ Normalized features (mean={X_norm.mean():.3f}, std={X_norm.std():.3f})")

    # Step 3: PCA Analysis
    print("\n3. PRINCIPAL COMPONENT ANALYSIS")
    print("-" * 40)

    pca, X_transformed = compute_pca(X_norm, n_components=50)
    print(f"✓ Computed PCA with {pca.n_components_} components")

    # Analyze eigenvalue spectrum
    eigen_analysis = analyze_eigenvalue_spectrum(pca)
    print(f"✓ Eigenvalue analysis:")
    print(f"  - Elbow at component: {eigen_analysis['elbow_index']}")
    print(f"  - Kaiser criterion suggests: {eigen_analysis['kaiser_n_components']} components")
    print(f"  - Components for 90% variance: {eigen_analysis['components_for_variance']['90%']}")
    print(f"  - Components for 95% variance: {eigen_analysis['components_for_variance']['95%']}")

    # Step 4: Stability Analysis
    print("\n4. STABILITY ANALYSIS ACROSS BATCHES")
    print("-" * 40)

    # Create batch dictionaries
    batch_data = create_batch_dict(df)
    batch_labels = create_label_dict(df)

    # Normalize each batch and compute PCA
    pca_list = []
    for batch_id in sorted(batch_data.keys()):
        X_batch = batch_data[batch_id]
        X_batch_norm, _ = normalize_features(X_batch, method='standard')
        pca_batch, _ = compute_pca(X_batch_norm, n_components=20)
        pca_list.append(pca_batch)

    # Measure PC stability
    stability_angles = measure_pc_stability(pca_list, n_components=10)
    mean_angles = stability_angles.mean(axis=0)

    print(f"✓ PC stability analysis (mean angles between batches):")
    for i, angle in enumerate(mean_angles[:5]):
        stability = "STABLE" if angle < 15 else "UNSTABLE" if angle > 30 else "MODERATE"
        print(f"  - PC{i+1}: {angle:.1f}° ({stability})")

    # Find stable subspace
    stable_components = find_stable_subspace(stability_angles, stability_threshold=20.0)
    print(f"✓ Stable components (< 20°): {[i+1 for i in stable_components]}")

    # Step 5: Drift Tracking
    print("\n5. DRIFT TRACKING IN PC SPACE")
    print("-" * 40)

    # Use first batch PCA as reference
    reference_pca = pca_list[0]

    # Track drift
    drift_df = track_drift_in_pc_space(batch_data, reference_pca, batch_labels)

    # Calculate average drift velocity
    avg_velocities = drift_df.groupby('chemical')['drift_velocity'].mean()
    print(f"✓ Average drift velocities by chemical:")
    for chem_id, velocity in avg_velocities.items():
        if chem_id != 'all':
            chem_name = df[df['chemical'] == chem_id]['chemical_name'].iloc[0]
            print(f"  - {chem_name}: {velocity:.3f} units/batch")

    # Step 6: Clustering Quality
    print("\n6. CLUSTERING QUALITY EVALUATION")
    print("-" * 40)

    # Prepare data for clustering evaluation
    X_list = []
    labels_list = []
    for batch_id in sorted(batch_data.keys()):
        X_batch = batch_data[batch_id]
        X_batch_norm, _ = normalize_features(X_batch, method='standard')
        X_batch_pca = reference_pca.transform(X_batch_norm)[:, :10]
        X_list.append(X_batch_pca)
        labels_list.append(batch_labels[batch_id])

    # Evaluate clustering stability
    silhouette_scores = evaluate_clustering_stability(X_list, labels_list, metric='silhouette')
    print(f"✓ Silhouette scores by batch:")
    for i, score in enumerate(silhouette_scores, 1):
        print(f"  - Batch {i}: {score:.3f}")

    # Compare different subspaces
    X_full = np.vstack(X_list)
    y_full = np.hstack(labels_list)

    subspace_results = compare_subspace_clustering(
        X_full,
        y_full,
        subspace_dims=[
            list(range(10)),           # All first 10 PCs
            stable_components[:5],      # Only stable PCs
            list(range(5, 10))         # Only unstable PCs
        ]
    )

    print(f"✓ Subspace comparison (silhouette scores):")
    print(f"  - All PCs (1-10): {subspace_results['subspace_0']['silhouette_score']:.3f}")
    print(f"  - Stable PCs only: {subspace_results['subspace_1']['silhouette_score']:.3f}")
    print(f"  - Unstable PCs only: {subspace_results['subspace_2']['silhouette_score']:.3f}")

    # Step 7: Drift Correction
    print("\n7. DRIFT CORRECTION")
    print("-" * 40)

    # Get first and last batch for comparison
    X_batch1 = batch_data[1]
    X_batch5 = batch_data[5]
    labels_batch1 = batch_labels[1]
    labels_batch5 = batch_labels[5]

    # Normalize
    X_batch1_norm, _ = normalize_features(X_batch1, method='standard')
    X_batch5_norm, _ = normalize_features(X_batch5, method='standard')

    # Measure alignment before correction
    distance_before = np.linalg.norm(X_batch1_norm.mean(axis=0) - X_batch5_norm.mean(axis=0))

    # Apply Procrustes correction
    X_batch5_corrected, transform = procrustes_alignment(X_batch5_norm, X_batch1_norm)

    # Measure alignment after correction
    distance_after = np.linalg.norm(X_batch1_norm.mean(axis=0) - X_batch5_corrected.mean(axis=0))

    print(f"✓ Procrustes alignment results:")
    print(f"  - Distance before correction: {distance_before:.3f}")
    print(f"  - Distance after correction: {distance_after:.3f}")
    print(f"  - Improvement: {(1 - distance_after/distance_before)*100:.1f}%")

    # Apply incremental correction to all batches
    batch_list = [batch_data[i] for i in sorted(batch_data.keys())]
    corrected_batches = incremental_drift_correction(batch_list, reference_idx=0)

    print(f"✓ Applied incremental drift correction to all {len(corrected_batches)} batches")

    # Step 8: Visualization
    print("\n8. GENERATING VISUALIZATIONS")
    print("-" * 40)

    # Create results directory
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # Plot 1: Eigenvalue spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    eigenvalues = eigen_analysis['eigenvalues'][:30]
    ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-')
    ax1.axvline(eigen_analysis['elbow_index'], color='r', linestyle='--', label='Elbow')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Eigenvalue Spectrum (Scree Plot)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    cumvar = eigen_analysis['cumulative_variance'][:30]
    ax2.plot(range(1, len(cumvar) + 1), cumvar, 'o-')
    ax2.axhline(0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(0.95, color='g', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/eigenvalue_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved eigenvalue analysis plot")

    # Plot 2: PC stability
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(stability_angles.T, cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel('Batch Transition')
    ax.set_ylabel('Principal Component')
    ax.set_title('PC Stability: Angular Changes Between Batches')
    ax.set_xticks(range(stability_angles.shape[0]))
    ax.set_xticklabels([f'{i+1}→{i+2}' for i in range(stability_angles.shape[0])])
    ax.set_yticks(range(stability_angles.shape[1]))
    ax.set_yticklabels([f'PC{i+1}' for i in range(stability_angles.shape[1])])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Angle (degrees)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(stability_angles.shape[0]):
        for j in range(stability_angles.shape[1]):
            text = ax.text(i, j, f'{stability_angles[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig('results/figures/pc_stability.png', dpi=150, bbox_inches='tight')
    print("✓ Saved PC stability heatmap")

    # Plot 3: Drift visualization in PC space
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette('husl', 6)  # 6 chemicals

    for batch_id in sorted(batch_data.keys()):
        X_batch = batch_data[batch_id]
        X_batch_norm, _ = normalize_features(X_batch, method='standard')
        X_batch_pca = reference_pca.transform(X_batch_norm)[:, :3]
        labels_batch = batch_labels[batch_id]

        for chem_id in range(6):
            mask = labels_batch == chem_id
            if mask.any():
                ax.scatter(X_batch_pca[mask, 0],
                          X_batch_pca[mask, 1],
                          X_batch_pca[mask, 2],
                          c=[colors[chem_id]],
                          alpha=min(0.3 + 0.1 * batch_id, 1.0),  # Fade with batch, cap at 1.0
                          s=20,
                          label=f'Chem {chem_id}, Batch {batch_id}' if batch_id == 1 else '')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Sensor Drift Visualization in First 3 PCs')
    # Only show legend for first batch
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:6], labels[:6], loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('results/figures/drift_3d_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved 3D drift visualization")

    plt.close('all')

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nAll modules tested:")
    print("✓ Synthetic data generation")
    print("✓ Data preprocessing")
    print("✓ PCA analysis")
    print("✓ Stability metrics")
    print("✓ Drift tracking")
    print("✓ Drift correction")
    print("✓ Visualizations")
    print("\nResults saved to: results/figures/")


if __name__ == "__main__":
    main()