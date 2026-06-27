# Supported Models

## Classification

| Model | Key Parameters | Best For |
|---|---|---|
| Random Forest | n_estimators, max_depth | General purpose, handles missing data |
| Gradient Boosting | n_estimators, learning_rate, max_depth | High accuracy, tabular data |
| Logistic Regression | C, max_iter | Binary classification, interpretable |
| Support Vector Machine | C, kernel | Small-medium datasets |

## Regression

| Model | Key Parameters | Best For |
|---|---|---|
| Random Forest | n_estimators, max_depth | Non-linear relationships |
| Gradient Boosting | n_estimators, learning_rate | High accuracy regression |
| Linear Regression | — | Linear relationships |
| Ridge Regression | alpha | Linear with regularization |
| Lasso Regression | alpha | Feature selection |
| Support Vector Regression | C, kernel | Non-linear regression |

## Clustering

| Model | Key Parameters | Best For |
|---|---|---|
| K-Means | n_clusters | Known number of clusters |
| DBSCAN | eps, min_samples | Unknown clusters, noise detection |

## Hyperparameter Tuning

| Method | Description |
|---|---|
| Grid Search | Exhaustive search over all combinations |
| Randomized Search | Random sampling — faster, good for large grids |

## Metrics

**Classification:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix

**Regression:** MSE, RMSE, MAE, R² Score

**Clustering:** Silhouette Score, Number of Clusters
