# Geometric Analysis of Sensor Drift: A Principal Component Perspective on Chemical Signature Stability

## 📋 Project Overview

This project investigates sensor drift in gas sensor arrays through unsupervised learning, specifically using **Principal Component Analysis (PCA)** as the primary dimensionality reduction technique. The analysis reframes the traditional calibration problem as a geometric analysis task: understanding how the low-dimensional manifold occupied by chemical signatures transforms over time in high-dimensional measurement space.

**Key Finding:** The project revealed a fundamental trade-off between chemical discrimination and temporal stability—high-variance principal components provide superior clustering quality because they capture chemical differences, independent of (and despite) their temporal instability.

## 🎯 Research Question

How does sensor drift affect the principal component structure of chemical signatures, and what is the relationship between variance explained, discriminative power, and temporal stability?

## 📊 Key Results

### 1. Dimensionality Discovery ✓
- **Finding:** Data lives in an **8-dimensional manifold** (90% variance explained)
- Achieved **94% compression** from 128 to 8 dimensions
- Kaiser criterion identified 12 components with eigenvalues > 1
- Clear elbow in scree plot at k=6-8 components

### 2. Stability-Variance Trade-off ✓
- **Critical Discovery:** High-variance PCs drift more but cluster better
  - **PC1-3** (75% variance): Silhouette = 0.537, high drift (unstable)
  - **PC7-10** (5% variance): Silhouette = 0.486, low drift (stable)
  - **PC1-8** (90% variance): Silhouette = 0.508, balanced compromise
- High-variance PCs drift **5.6× more** than low-variance PCs
- Variance explained and discriminative power are **orthogonal** to temporal stability

### 3. Drift Characteristics ✓
- **Linear drift patterns** with r=0.986 correlation (predictable, not random)
- Drift manifests as **rigid transformations** (rotation/translation) preserving cluster structure
- Directional drift trajectory in PC1-PC2 space across 10 batches (36 months)
- Within-batch dispersion remains stable despite centroid shifts

### 4. Clustering Performance ✓
- **Optimal k=6** clusters identified (matching 6 gas types)
- Chemical signatures remain **distinguishable despite drift**
- PC1-3 maintains excellent cluster separation across all time points
- Cluster identity vs. cluster separation trade-off discovered

### 5. Separation vs. Identity Discovery ✓
**The most important finding:**
- **PC1-3 (Unstable):** Excellent separation but >70° rotation → cluster identity lost
- **PC7-10 (Stable):** Poor separation but <40° rotation → cluster identity preserved
- For unsupervised deployment, knowing "6 well-separated groups exist" is useless without knowing which group = which gas

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

## 🔬 Methodology

### Unsupervised Learning Approach
- **Primary Algorithm:** Principal Component Analysis (PCA)
  - Dimensionality reduction from 128D to interpretable subspaces
  - Global PCA across all batches for consistent reference frame
  - Variance analysis and eigenvalue decomposition

- **Validation:** K-means Clustering
  - Optimal k=6 determined via elbow method and silhouette analysis
  - Clustering quality metrics: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score
  - Multi-algorithm consistency testing across PC subspaces

### Analysis Framework
1. **Data Quality Assessment:** Zero missing values, no duplicates, 13,910 samples across 10 batches
2. **Exploratory Data Analysis:** Correlation structure, temporal patterns, variance distribution
3. **PCA-Based Drift Analysis:** Centroid tracking, stability metrics, geometric transformations
4. **Multi-Algorithm Validation:** Clustering consistency across different PC representations

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

## 💡 Key Insights & Implications

### What We Learned
1. **Dimensionality:** Despite 128 sensors, chemical signatures occupy an ~8D manifold
2. **Variance ≠ Stability:** High-variance PCs capture chemical differences but drift more over time
3. **Geometric Drift:** Drift is structured (linear, directional) not random noise
4. **Discrimination-Stability Trade-off:** Cannot maximize both simultaneously in unsupervised setting

### Practical Implications

**For Sensor Array Deployment:**
- **High-Accuracy Applications:** Use PC1-8 with periodic recalibration
- **Maintenance-Limited Deployments:** Use PC7-10 for stable (if less discriminative) performance
- **Balanced Approach:** PC1-8 provides 90% variance with acceptable drift characteristics

**What Unsupervised Analysis Can Determine:**
- ✓ Drift exists and is systematic
- ✓ High-variance PCs drift more than low-variance PCs
- ✓ Clustering remains effective despite drift
- ✓ Data occupies low-dimensional manifold

**What Requires Supervised Information:**
- ✗ Root cause (sensor degradation vs. environmental changes)
- ✗ Which specific sensors are failing
- ✗ Optimal correction strategy
- ✗ Whether drift affects chemical identity or only measurement geometry

### Limitations
- Linear assumptions (PCA may miss nonlinear patterns)
- Cannot distinguish sensor degradation from environmental sensitivity without controlled experiments
- Results specific to metal-oxide sensors over 36 months
- Unsupervised constraint prevents implementing advanced correction (e.g., Procrustes alignment)

## 🚀 Future Work

### Unsupervised Drift Correction (No Labels Required)
1. **Temporal cluster tracking:** Hungarian algorithm to match centroids across batches
2. **Incremental PCA updating:** Online algorithms for adaptive reference frames
3. **Drift velocity modeling:** Linear drift models (r=0.986) for predictive correction
4. **Adaptive weighted PCA:** Dynamic weighting based on cumulative drift

### Extensions
- **Nonlinear methods:** Kernel PCA, autoencoders, UMAP to capture nonlinear drift patterns
- **Physical investigation:** Loading analysis to identify which sensors contribute to drifting PCs
- **Cross-validation:** Test on other sensor array types (electrochemical, optical)

## 📁 Project Structure

```
sensor-drift-pca/
├── data/
│   ├── raw/                    # Original UCI dataset files (10 batches)
│   └── processed/              # Cleaned CSV format (sensor_data.csv)
├── notebooks/
│   ├── main.ipynb             # Complete analysis notebook
│   └── out/
│       └── main.pdf           # Full project report (42 pages)
├── src/
│   └── sensor_drift/
│       └── process_dat_to_csv.py  # LibSVM to CSV converter
├── report/                     # LaTeX report source
├── Presentation/
│   └── presentation.pdf       # Project presentation
└── README.md                  # This file
```

## 🔗 References

- Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012). "Chemical Gas Sensor Drift Compensation Using Classifier Ensembles." *Sensors and Actuators B: Chemical* 166: 320-29. https://doi.org/10.1016/j.snb.2012.01.074

- Rodriguez-Lujan, I., Fonollosa, J., Vergara, A., Homer, M., & Huerta, R. (2014). "On the Calibration of Sensor Arrays for Pattern Recognition Using the Minimal Number of Experiments." *Chemometrics and Intelligent Laboratory Systems* 130: 123-34. https://doi.org/10.1016/j.chemolab.2013.10.012

- UCI Machine Learning Repository: [Gas Sensor Array Drift Dataset](https://doi.org/10.24432/C5ZS4K)

## 📄 License

This project is for academic purposes. Dataset courtesy of UCI Machine Learning Repository.

## 👤 Author

**Thomas Fey-Grytnes**
University of Colorado Boulder
Course: Artificial Intelligence - Unsupervised Learning

---

*Project completed with comprehensive analysis of 13,910 sensor measurements across 36 months, revealing fundamental insights about the geometry of sensor drift in high-dimensional spaces.*
