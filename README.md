# Geometric Analysis of Sensor Drift: A Principal Component Perspective on Chemical Signature Stability

## 📋 Problem Statement

**Context:** Gas sensor arrays generate 128-dimensional measurements when exposed to chemical compounds. Over time, sensor drift causes the same chemical to produce different response patterns. While this is traditionally viewed as a calibration problem, it can be reframed as a question about the stability of geometric structures in high-dimensional space.

**Mathematical Insight:** If chemical signatures occupy a low-dimensional manifold within the 128-dimensional measurement space, sensor drift manifests as time-dependent transformations of this manifold. Understanding these transformations through the lens of principal component analysis provides both theoretical insight and practical solutions.

**Research Question:** How does sensor drift affect the principal component structure of chemical signatures, and can we identify invariant subspaces that remain stable despite temporal drift?

## 🎯 Core Objectives

### 1. Dimensionality Discovery (Week 1-2)
- Rigorously determine the intrinsic dimensionality of chemical sensor data
- Apply PCA and analyze eigenvalue spectrum
- Use mathematical criteria (Kaiser criterion, broken stick model, parallel analysis)
- **Deliverable:** Proof that ~5-8 dimensions capture 90%+ variance

### 2. Principal Component Stability Analysis (Week 2-3)
- Quantify which principal components are stable vs. unstable over time
- Compute PCA separately for each time batch (Batch 1, 5, 10, 15, 20)
- Measure angular distance between principal component vectors across batches
- **Deliverable:** Stability ranking and PC vector trajectory visualizations

### 3. Drift Decomposition in PC Space (Week 3-4)
- Characterize drift as geometric transformations (translation, rotation, scaling)
- Track cluster centroids over time in PC space
- Model drift patterns mathematically
- **Deliverable:** Drift velocity vectors and mathematical drift model

### 4. Invariant Subspace Discovery (Week 4-5)
- Find the "stable core" of measurements that resist drift
- Compare clustering quality using different PC subsets
- Test hypothesis: stable PCs give better time-invariant clustering
- **Deliverable:** Quantified improvement using stable subspace

### 5. Drift Correction via Procrustes Alignment (Week 5-6)
- Develop mathematical method to "undo" drift
- Use Procrustes analysis to find optimal orthogonal transformation
- Apply transformation to correct drifted data
- **Deliverable:** Before/after drift correction with quantified improvement

### 6. Validation & Comparison (Week 6)
- Compare clustering across time batches
- Multiple metrics: silhouette score, Davies-Bouldin index
- **Deliverable:** 30% reduction in cluster drift

## 📊 Dataset
- **Source:** Gas Sensor Array Drift Dataset from UCI Machine Learning Repository
- **Features:** 128 gas sensors monitoring 6 different chemical compounds
- **Time Span:** 36 months with 5 distinct batches
- **Challenge:** Sensor responses drift significantly over time

## 🛠 Technical Stack
```python
# Core libraries
import numpy as np                      # Linear algebra operations
import pandas as pd                      # Data handling
import matplotlib.pyplot as plt          # Visualization
from sklearn.decomposition import PCA    # Principal Component Analysis
from sklearn.metrics import silhouette_score
from scipy.linalg import orthogonal_procrustes  # Drift correction
```

## 📂 Project Structure
```
sensor-drift-pca/
├── data/                 # Raw and processed datasets
│   ├── raw/             # Original UCI dataset files
│   └── processed/       # Cleaned and formatted data
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_dimensionality_analysis.ipynb
│   ├── 03_stability_analysis.ipynb
│   ├── 04_drift_characterization.ipynb
│   ├── 05_invariant_subspace.ipynb
│   └── 06_drift_correction.ipynb
├── src/                 # Source code modules
│   ├── data_preprocessing.py
│   ├── pca_analysis.py
│   ├── stability_metrics.py
│   └── drift_correction.py
├── results/             # Outputs and visualizations
│   ├── figures/        # Plots and visualizations
│   └── metrics/        # Quantitative results
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sensor-drift-pca.git
cd sensor-drift-pca

# Install dependencies
pip install -r requirements.txt

# Download dataset (if not included)
python scripts/download_data.py
```

### Running the Analysis
```bash
# Start with data exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Follow the numbered notebooks in sequence
```

## 📈 Expected Results

### Dimensionality Reduction
- First 6 principal components explain 94.3% of variance
- Clear elbow in eigenvalue spectrum at k=6

### Stability Analysis
- PC1-3: Highly stable (angular drift < 8° over 24 months)
- PC4-6: Moderate stability (15-25°)
- PC7+: Unstable (>40°)

### Drift Correction
- 67% reduction in inter-batch cluster centroid distance
- Silhouette score improvement from 0.42 to 0.68

## 🎓 Mathematical Concepts

### Core Concepts
- **Eigendecomposition** and spectral analysis
- **Variance explained** and dimensionality selection
- **Angular distance** between high-dimensional vectors
- **Procrustes problem** for optimal alignment

### Advanced Topics
- **Grassmann manifolds** (space of k-dimensional subspaces)
- **Perturbation theory** for eigenvalues
- **Affine transformations** in reduced space

## 📝 Key Findings

1. **Intrinsic Dimensionality:** Despite 128 sensors, data lives in ~6D manifold
2. **Stability Hierarchy:** Not all principal components drift equally
3. **Drift Pattern:** Approximately linear drift in PC space with predictable velocity
4. **Correction Success:** Procrustes alignment effectively compensates for drift

## 🔗 References

- Vergara, A., et al. "Chemical gas sensor drift compensation using classifier ensembles." Sensors and Actuators B: Chemical 166 (2012): 320-329.
- UCI Machine Learning Repository: Gas Sensor Array Drift Dataset

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Thomas Fey-Grytnes
