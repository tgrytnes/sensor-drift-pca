#!/usr/bin/env python3
"""Baseline training/evaluation to gauge model viability under sensor drift."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler

from pca_analysis import infer_sensor_columns
from data_preprocessing import normalize_features
from drift_correction import correct_batch_drift
from scipy.linalg import orthogonal_procrustes
import warnings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    project_root = Path(__file__).resolve().parents[2]
    default_data = project_root / 'data' / 'processed' / 'sensor_data.csv'
    default_output = project_root / 'results' / 'model_eval'

    parser = argparse.ArgumentParser(description="Train/evaluate baseline classifier")
    parser.add_argument('--data', type=Path, default=default_data,
                        help='Path to processed sensor dataset CSV')
    parser.add_argument('--label-col', default='gas_name',
                        help='Column containing target labels')
    parser.add_argument('--batch-col', default='batch',
                        help='Column containing batch identifiers')
    parser.add_argument('--train-batches', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='Batch IDs to use for training')
    parser.add_argument('--test-batches', type=int, nargs='+', default=[7, 8, 9, 10],
                        help='Batch IDs to reserve for evaluation')
    parser.add_argument('--n-components', type=int, default=30,
                        help='Upper bound on PCA components to fit')
    parser.add_argument('--stable-threshold', type=float, default=0.5,
                        help='Std threshold (across training batches) for stable PCs')
    parser.add_argument('--model', choices=['logistic', 'random_forest', 'hist_gradient_boost', 'svc', 'ensemble'],
                        default='logistic', help='Classifier type to train')
    parser.add_argument('--drift-align', choices=['none', 'mean_shift', 'procrustes', 'mean_shift_procrustes'], default='none',
                        help='Apply specified drift correction to test batches before normalization')
    parser.add_argument('--oversample', action='store_true', help='Oversample minority classes in the training set')
    parser.add_argument('--output-dir', type=Path, default=default_output,
                        help='Directory to store evaluation artifacts')
    parser.add_argument('--class-weight-override', default='',
                        help='Comma-separated overrides like Acetaldehyde=2.0,Toluene=1.3')
    parser.add_argument('--ensemble-weight', type=float, default=0.5,
                        help='Weight for logistic in ensemble (0.0-1.0)')
    parser.add_argument('--no-save', action='store_true', help='Skip writing artifacts to disk')

    return parser.parse_args()

def identify_stable_components(scores: np.ndarray,
                               batches: np.ndarray,
                               batch_ids: Sequence[int],
                               threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of PCs whose batch-wise means stay below threshold."""
    batch_means = []
    for batch_id in batch_ids:
        mask = batches == batch_id
        if mask.sum() == 0:
            continue
        batch_means.append(scores[mask].mean(axis=0))

    if not batch_means:
        return np.array([], dtype=int), np.array([])

    batch_means = np.vstack(batch_means)
    std_by_component = batch_means.std(axis=0)
    stable_idx = np.where(std_by_component < threshold)[0]

    if stable_idx.size == 0:
        # Fall back to five least-variant components so we always have features.
        stable_idx = np.argsort(std_by_component)[:5]

    return stable_idx, std_by_component


def parse_class_weight_override(spec: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    if not spec:
        return overrides

    for pair in spec.split(','):
        if not pair.strip():
            continue
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        try:
            overrides[key] = float(value)
        except ValueError:
            continue
    return overrides


def train_and_evaluate(df: pd.DataFrame,
                       sensor_cols: List[str],
                       label_col: str,
                       batch_col: str,
                       train_batches: Sequence[int],
                       test_batches: Sequence[int],
                       n_components: int,
                       stable_threshold: float,
                       model: str = 'logistic',
                       drift_align: str = 'none',
                       oversample: bool = False,
                       class_weight_override: Dict[str, float] = None,
                       ensemble_weight: float = 0.5) -> Dict:
    """Fit PCA + logistic regression and collect evaluation metrics."""
    train_mask = df[batch_col].isin(train_batches)
    test_mask = df[batch_col].isin(test_batches)

    if not train_mask.any() or not test_mask.any():
        raise ValueError('Train/test split produced empty dataset.')

    train_df = df.loc[train_mask, sensor_cols + [label_col, batch_col]].copy()

    if oversample:
        target = train_df[label_col].value_counts().max()
        oversampled_groups = []
        minority_targets = {'Acetaldehyde'}
        for cls, group in train_df.groupby(label_col):
            if cls in minority_targets and len(group) < target:
                extra = group.sample(target - len(group), replace=True, random_state=42)
                group = pd.concat([group, extra], ignore_index=True)
            oversampled_groups.append(group)
        train_df = pd.concat(oversampled_groups, ignore_index=True)

    X_train_raw = train_df[sensor_cols].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy()
    train_batch_vector = train_df[batch_col].to_numpy()

    X_test_raw = df.loc[test_mask, sensor_cols].to_numpy(dtype=float)
    y_test = df.loc[test_mask, label_col].to_numpy()
    test_batch_vector = df.loc[test_mask, batch_col].to_numpy()

    drift_alignment_used = 'none'
    rng = np.random.default_rng(42)

    ensemble_weight = min(max(ensemble_weight, 0.0), 1.0)

    if drift_align in ('mean_shift', 'mean_shift_procrustes'):
        drift_alignment_used = 'mean_shift'
        aligned = X_test_raw.copy()
        for batch_id in test_batches:
            mask = test_batch_vector == batch_id
            if not mask.any():
                continue
            aligned[mask] = correct_batch_drift(
                X_test_raw[mask], X_train_raw, method='mean_shift'
            )
        X_test_raw = aligned

    elif drift_align == 'procrustes':
        drift_alignment_used = 'procrustes'

    if drift_align == 'mean_shift_procrustes':
        drift_alignment_used = 'mean_shift_procrustes'

    X_train_norm, norm_params = normalize_features(X_train_raw, method='standard')
    train_mean = norm_params['mean']
    train_std = norm_params['std']
    X_test_norm = (X_test_raw - train_mean) / (train_std + 1e-8)

    n_components = min(n_components, X_train_norm.shape[0], X_train_norm.shape[1])
    pca = PCA(n_components=n_components, svd_solver='full')
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        train_scores = pca.fit_transform(X_train_norm)
    if not np.isfinite(train_scores).all():
        raise ValueError('Non-finite values encountered in PCA training scores.')

    stable_idx, batch_std = identify_stable_components(
        train_scores, train_batch_vector, train_batches, stable_threshold)

    X_train_stable = train_scores[:, stable_idx]

    if drift_align in ('procrustes', 'mean_shift_procrustes') and len(stable_idx) > 0:
        align_components = min(len(stable_idx), 15, pca.n_components_)
        aligned = X_test_norm.copy()
        for batch_id in test_batches:
            mask = test_batch_vector == batch_id
            if not mask.any():
                continue
            batch_data = X_test_norm[mask]
            try:
                pca_batch = PCA(n_components=pca.n_components_, svd_solver='full')
                pca_batch.fit(batch_data)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    R, _ = orthogonal_procrustes(
                        pca_batch.components_[:align_components].T,
                        pca.components_[:align_components].T
                    )
                if not np.isfinite(R).all():
                    raise ValueError('Non-finite rotation matrix')
                aligned[mask] = batch_data @ R
            except Exception:
                aligned[mask] = batch_data
        X_test_norm = aligned

    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        X_test_scores = pca.transform(X_test_norm)
    if not np.isfinite(X_test_scores).all():
        raise ValueError('Non-finite values encountered in PCA test scores.')
    X_test_stable = X_test_scores[:, stable_idx]

    class_labels = np.unique(y_train)
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    logistic_clf = None
    logistic_probs_aligned = None
    hist_clf = None
    hist_probs_aligned = None
    classes_order = None

    def align_probabilities(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
        aligned = np.zeros((probs.shape[0], len(class_labels)))
        for idx, cls in enumerate(classes):
            aligned[:, label_to_index[cls]] = probs[:, idx]
        return aligned

    if model in ('logistic', 'ensemble'):
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_stable)
        X_test_final = scaler.transform(X_test_stable)

        if oversample:
            class_weight = None
        else:
            counts = {label: (y_train == label).sum() for label in class_labels}
            total = len(y_train)
            class_weight = {label: total / (len(class_labels) * count)
                            for label, count in counts.items()}
            if class_weight_override:
                for cls, factor in class_weight_override.items():
                    if cls in class_weight:
                        class_weight[cls] *= factor

        logistic_clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
        old_err = np.seterr(over='ignore', under='ignore', divide='ignore', invalid='ignore')
        try:
            logistic_clf.fit(X_train_final, y_train)
        finally:
            np.seterr(**old_err)

        logistic_probs = logistic_clf.predict_proba(X_test_final)
        logistic_probs_aligned = align_probabilities(logistic_probs, logistic_clf.classes_)

        if model == 'logistic':
            y_pred = logistic_clf.predict(X_test_final)
            classes_order = logistic_clf.classes_

    if model == 'random_forest':
        rf_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        )
        rf_clf.fit(X_train_stable, y_train)
        y_pred = rf_clf.predict(X_test_stable)
        classes_order = rf_clf.classes_

    if model in ('hist_gradient_boost', 'ensemble'):
        counts = {label: (y_train == label).sum() for label in class_labels}
        class_to_weight = {label: len(y_train) / (len(class_labels) * count)
                           for label, count in counts.items()}
        sample_weights = np.array([class_to_weight[label] for label in y_train])
        if class_weight_override:
            sample_weights *= np.array([class_weight_override.get(label, 1.0) for label in y_train])

        hist_clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            l2_regularization=0.1,
            random_state=42
        )
        hist_clf.fit(X_train_stable, y_train, sample_weight=sample_weights)
        hist_probs = hist_clf.predict_proba(X_test_stable)
        hist_probs_aligned = align_probabilities(hist_probs, hist_clf.classes_)

        if model == 'hist_gradient_boost':
            y_pred = hist_clf.predict(X_test_stable)
            classes_order = hist_clf.classes_

    if model == 'svc':
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_stable)
        X_test_final = scaler.transform(X_test_stable)

        svc_clf = SVC(kernel='rbf', C=3.0, gamma='scale', class_weight='balanced')
        svc_clf.fit(X_train_final, y_train)
        y_pred = svc_clf.predict(X_test_final)
        classes_order = svc_clf.classes_

    if model == 'ensemble':
        if logistic_probs_aligned is None or hist_probs_aligned is None:
            raise ValueError('Ensemble requires both logistic and hist_gradient_boost components')
        combined_probs = ensemble_weight * logistic_probs_aligned + (1.0 - ensemble_weight) * hist_probs_aligned
        y_indices = np.argmax(combined_probs, axis=1)
        y_pred = class_labels[y_indices]
        classes_order = class_labels

    if classes_order is None:
        raise ValueError(f"Unsupported model: {model}")

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes_order)

    per_batch = {}
    for batch_id in test_batches:
        mask = df.loc[test_mask, batch_col].to_numpy() == batch_id
        if mask.sum() == 0:
            continue
        per_batch[str(batch_id)] = float(accuracy_score(y_test[mask], y_pred[mask]))

    return {
        'model': model,
        'drift_alignment': drift_alignment_used,
        'oversampled': oversample,
        'class_weight_override': class_weight_override or {},
        'ensemble_weight': ensemble_weight,
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'classes': list(classes_order),
        'stable_components': (stable_idx + 1).tolist(),  # 1-indexed for readability
        'stable_threshold': float(stable_threshold),
        'batch_std': batch_std.tolist(),
        'per_batch_accuracy': per_batch,
        'n_train_samples': int(len(X_train_raw)),
        'n_test_samples': int(test_mask.sum()),
        'n_features_used': int(len(stable_idx))
    }


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f'Dataset not found at {args.data}')

    df = pd.read_csv(args.data)
    sensor_cols = infer_sensor_columns(df)

    metrics = train_and_evaluate(
        df=df,
        sensor_cols=sensor_cols,
        label_col=args.label_col,
        batch_col=args.batch_col,
        train_batches=args.train_batches,
        test_batches=args.test_batches,
        n_components=args.n_components,
        stable_threshold=args.stable_threshold,
        model=args.model,
        drift_align=args.drift_align,
        oversample=args.oversample,
        class_weight_override=parse_class_weight_override(args.class_weight_override),
        ensemble_weight=args.ensemble_weight
    )

    print('=' * 70)
    print('BASELINE DRIFT-ROBUST CLASSIFIER')
    print(f"Train batches: {args.train_batches} | Test batches: {args.test_batches}")
    print(f"Samples: train={metrics['n_train_samples']} test={metrics['n_test_samples']}")
    print(f"Model: {metrics['model']} | Drift alignment: {metrics['drift_alignment']} | Oversampled: {metrics['oversampled']} | Ensemble weight: {metrics['ensemble_weight']:.2f}")
    print(f"Stable PCs used: {metrics['stable_components']} (threshold={metrics['stable_threshold']})")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print('Per-batch accuracy:')
    for batch_id, value in metrics['per_batch_accuracy'].items():
        print(f"  Batch {batch_id}: {value:.4f}")
    print('=' * 70)

    if not args.no_save:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / 'baseline_metrics.json'
        metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
