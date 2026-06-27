# Changelog

All notable changes to MLOps Platform are documented here.

---

## [v1.0.0] — 2026-06-27

### Added
- 9-page Streamlit UI with dark theme
- Experiment tracking with SQLite backend
- Multi-model training: Random Forest, Gradient Boosting, Logistic Regression, SVM, Ridge, Lasso, K-Means, DBSCAN
- Grid Search and Randomized Search hyperparameter tuning
- 5-fold cross-validation with stratified splitting
- Model registry with versioning and stage promotion (staging / production / champion / archived)
- Data drift detection using KS-test and Population Stability Index (PSI)
- Automated retraining jobs triggered on drift detection
- Interactive Plotly dashboards: run timeline, metric trends, confusion matrix, feature importance
- Pipeline audit log with component-level filtering and download
- 45 pytest unit + integration tests
- GitHub Actions CI/CD workflow
- 7 built-in datasets (Iris, Wine, Breast Cancer, Diabetes, Synthetic x3)
- CSV upload support for custom datasets
- PDF and JSON export for analysis results
- Correlation matrix and feature distribution explorer
