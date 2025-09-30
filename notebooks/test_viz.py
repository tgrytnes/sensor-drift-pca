#!/usr/bin/env python3
"""
Simple test script to generate visualizations inline.
Run this to test if the visualization code works, then copy to notebook.
"""

import sys
import os
sys.path.append('../src')

# Change to notebook directory
os.chdir('/Users/thomasfey-grytnes/Documents/Artificial Intelligence - Studying/Hotel_Cancellation_Risk/notebooks')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Load experiment results manually
def load_experiment_results_from_dir(artifacts_dir, model_name):
    """Load predictions, metrics, and feature importance from experiment directory."""
    results = {'name': model_name}

    # Load predictions
    pred_path = artifacts_dir / "predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            results['predictions'] = json.load(f)
        print(f"✓ Loaded predictions from {pred_path}")

    # Load metrics
    metrics_path = artifacts_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results['metrics'] = json.load(f)
        print(f"✓ Loaded metrics from {metrics_path}")

    # Load feature importance
    importance_path = artifacts_dir / "feature_importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            results['importance'] = json.load(f)
        print(f"✓ Loaded importance from {importance_path}")

    return results

# Load results from all three models
base_dir = Path("..")
experiments = {}

artifact_dirs = {
    'rf_300': base_dir / 'artifacts_rf_300',
    'l2_moderate': base_dir / 'artifacts_l2_moderate',
    'xgb_lr1_d5': base_dir / 'artifacts_xgb_lr1_d5'
}

print("Loading experiment results...")
for exp_name, artifacts_dir in artifact_dirs.items():
    if artifacts_dir.exists():
        experiments[exp_name] = load_experiment_results_from_dir(artifacts_dir, exp_name)
        print(f"✓ Loaded {exp_name} model results")
    else:
        print(f"✗ Directory not found: {artifacts_dir}")

if not experiments:
    print("❌ No experiments loaded!")
    sys.exit(1)

print(f"\n📊 Ready to generate visualizations for {len(experiments)} models...")

# Test 1: Simple confusion matrix
print("\n1️⃣ Testing Confusion Matrix...")
from sklearn.metrics import confusion_matrix

# Create confusion matrices subplot
n_models = len(experiments)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
if n_models == 1:
    axes = [axes]

for idx, (exp_name, exp_data) in enumerate(experiments.items()):
    if 'predictions' not in exp_data:
        print(f"⚠️ No predictions for {exp_name}")
        continue

    preds = exp_data['predictions']
    y_true = np.array(preds['y_true'])
    y_pred = np.array(preds['y_pred'])

    print(f"   {exp_name}: {len(y_true)} predictions loaded")

    cm = confusion_matrix(y_true, y_pred)
    ax = axes[idx]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Clean up model name
    display_name = exp_name.replace('_', ' ').title()
    if 'Rf' in display_name:
        display_name = display_name.replace('Rf', 'Random Forest')
    if 'L2' in display_name:
        display_name = display_name.replace('L2', 'LogReg L2')
    if 'Xgb' in display_name:
        display_name = display_name.replace('Xgb', 'XGBoost')

    ax.set_title(f'{display_name}\nConfusion Matrix', fontsize=12)

    # Labels
    classes = ['Not Canceled', 'Canceled']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

plt.tight_layout()
plt.savefig('../artifacts/figures/test_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrices saved to ../artifacts/figures/test_confusion_matrices.png")

# Test 2: ROC Curves
print("\n2️⃣ Testing ROC Curves...")
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (exp_name, exp_data) in enumerate(experiments.items()):
    if 'predictions' not in exp_data:
        continue

    preds = exp_data['predictions']
    y_true = np.array(preds['y_true'])
    y_prob = np.array(preds['y_prob'])

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Clean up model name
    display_name = exp_name.replace('_', ' ').title()
    if 'Rf' in display_name:
        display_name = display_name.replace('Rf', 'Random Forest')
    if 'L2' in display_name:
        display_name = display_name.replace('L2', 'LogReg L2')
    if 'Xgb' in display_name:
        display_name = display_name.replace('Xgb', 'XGBoost')

    color = colors[idx % len(colors)]
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{display_name} (AUC = {roc_auc:.3f})')

# Plot random classifier line
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Hotel Cancellation Prediction\nModel Comparison', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('../artifacts/figures/test_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ ROC curves saved to ../artifacts/figures/test_roc_curves.png")

print("\n🎉 Test visualizations completed!")
print("📁 Check ../artifacts/figures/ for:")
print("   - test_confusion_matrices.png")
print("   - test_roc_curves.png")
print("\nIf these files look good, the visualization code is working!")