# Model Improvement Notes

## Highlights
- Introduced PCA-space Procrustes alignment plus class weighting for logistic regression; achieved balanced accuracy ≈0.70.
- Added ensemble mode (logistic + histogram gradient boosting) with weighted blending; best run reached balanced accuracy 0.736 using 30 PCs, Procrustes alignment, and class weights `{Acetaldehyde: 2.0, Toluene: 1.3}`.
- Batch-level performance improved markedly: batches 7–10 at roughly 0.84 / 0.84 / 0.66 / 0.63 accuracy.
- Acetaldehyde recall rose from ~0.15 to ~0.37; Toluene recall reached ~0.56 without hurting other classes.

## Experiments Tried
- Mean-shift alignment alone (moderate gains) vs. PCA Procrustes (major boost); combining both offered no extra benefit.
- Stratified oversampling for minority classes: helped more when used with class weights, but simple oversample-only harmed batch 9.
- Class-weight overrides (Acetaldehyde=2.0, Toluene=1.3) were more effective than data duplication.
- Ensemble weight sweep (0.5 → 0.6) traded some overall accuracy for higher batch-8 stability; best compromise at 0.6.

## Struggles / Caveats
- Procrustes alignment on raw sensor space was numerically unstable (overflow warnings, accuracy collapse); restricting to PCA space fixed it.
- Increasing ensemble weight towards logistic improves batch 8 but can lower overall recall; careful tuning required if specific batches need priority.
- Python warnings from SciPy SVD appear during alignment; they are now suppressed, but further smoothing (e.g., ridge regularisation) might be needed for production.

## Suggested Next Steps
- Automate a small sweep over ensemble weights (0.55–0.65) and class-weight tweaks to find the Pareto frontier between batch 8 accuracy and overall balance.
- Explore incremental fine-tuning using early samples from each test batch to push balanced accuracy beyond 0.74.
- Integrate the saved metrics (`results/model_eval/baseline_metrics.json`) into the notebook/report for documentation.


## Unsupervised Analyses Added
- Added notebook section that summarises explained variance, PC rotation angles, and visualises batch trajectories in original vs Procrustes-aligned PC space.
- Computed dispersion metrics before/after alignment showing reduced centroid radius after Procrustes correction.
- Implemented K-means and DBSCAN clustering diagnostics to compare silhouette/Davies–Bouldin scores and centroid drift across batches.
