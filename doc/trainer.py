"""
app/trainer.py — ML Training Engine for MLOps Platform
Supports: Classification, Regression, Clustering
Models: RandomForest, GradientBoosting, LogisticRegression, SVM, LinearRegression, Ridge, KMeans
Features: Preprocessing pipeline, cross-validation, hyperparameter search, feature importance
"""

import time, json, logging, numpy as np, pandas as pd
from pathlib import Path
from typing import Any

from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.impute           import SimpleImputer
from sklearn.model_selection  import (cross_val_score, GridSearchCV,
                                      RandomizedSearchCV, train_test_split,
                                      StratifiedKFold, KFold)
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, confusion_matrix,
                                      mean_squared_error, mean_absolute_error, r2_score,
                                      silhouette_score, classification_report)
from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble         import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model     import LogisticRegression, Ridge, LinearRegression, Lasso
from sklearn.svm              import SVC, SVR
from sklearn.cluster          import KMeans, DBSCAN
from sklearn.datasets         import (make_classification, make_regression,
                                      load_iris, load_wine, load_breast_cancer,
                                      load_diabetes, make_blobs)
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
CLASSIFICATION_MODELS = {
    "Random Forest":         RandomForestClassifier,
    "Gradient Boosting":     GradientBoostingClassifier,
    "Logistic Regression":   LogisticRegression,
    "Support Vector Machine":SVC,
}
REGRESSION_MODELS = {
    "Random Forest":         RandomForestRegressor,
    "Gradient Boosting":     GradientBoostingRegressor,
    "Linear Regression":     LinearRegression,
    "Ridge Regression":      Ridge,
    "Lasso Regression":      Lasso,
    "Support Vector Machine":SVR,
}
CLUSTERING_MODELS = {
    "K-Means": KMeans,
    "DBSCAN":  DBSCAN,
}

DEFAULT_PARAMS = {
    "Random Forest":          {"n_estimators":100,"max_depth":None,"min_samples_split":2,"random_state":42},
    "Gradient Boosting":      {"n_estimators":100,"learning_rate":0.1,"max_depth":3,"random_state":42},
    "Logistic Regression":    {"C":1.0,"max_iter":1000,"random_state":42},
    "Support Vector Machine": {"C":1.0,"kernel":"rbf","random_state":42},
    "Linear Regression":      {},
    "Ridge Regression":       {"alpha":1.0},
    "Lasso Regression":       {"alpha":1.0},
    "K-Means":                {"n_clusters":3,"random_state":42,"n_init":10},
    "DBSCAN":                 {"eps":0.5,"min_samples":5},
}

PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators":    [50,100,200],
        "model__max_depth":       [None,5,10,20],
        "model__min_samples_split":[2,5,10],
    },
    "Gradient Boosting": {
        "model__n_estimators":  [50,100,200],
        "model__learning_rate": [0.01,0.1,0.2],
        "model__max_depth":     [3,5,7],
    },
    "Logistic Regression": {
        "model__C":       [0.01,0.1,1.0,10.0],
        "model__max_iter":[500,1000],
    },
    "Ridge Regression": {
        "model__alpha": [0.01,0.1,1.0,10.0,100.0],
    },
}


# ── Dataset generators ────────────────────────────────────────────────────────
def get_builtin_dataset(name: str) -> tuple[pd.DataFrame, str, str]:
    """Returns (df, target_column, task_type)"""
    if name == "Iris (Classification)":
        d = load_iris(as_frame=True)
        df = d.frame
        df["target"] = d.target
        return df, "target", "classification"

    elif name == "Wine Quality (Classification)":
        d = load_wine(as_frame=True)
        df = d.frame
        df["target"] = d.target
        return df, "target", "classification"

    elif name == "Breast Cancer (Classification)":
        d = load_breast_cancer(as_frame=True)
        df = d.frame
        df["target"] = d.target
        return df, "target", "classification"

    elif name == "Diabetes (Regression)":
        d = load_diabetes(as_frame=True)
        df = d.frame
        return df, "target", "regression"

    elif name == "Synthetic Classification":
        X, y = make_classification(n_samples=1000, n_features=20,
                                   n_informative=10, n_redundant=5,
                                   n_classes=3, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        df["target"] = y
        return df, "target", "classification"

    elif name == "Synthetic Regression":
        X, y = make_regression(n_samples=1000, n_features=15,
                               n_informative=10, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
        df["target"] = y
        return df, "target", "regression"

    elif name == "Synthetic Clustering":
        X, y = make_blobs(n_samples=500, centers=4, n_features=8, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        df["cluster_label"] = y
        return df, "cluster_label", "clustering"

    else:
        raise ValueError(f"Unknown dataset: {name}")


def compute_dataset_stats(df: pd.DataFrame, target_col: str) -> dict:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {
        "shape":       list(df.shape),
        "missing":     int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
    }
    if target_col in df.columns:
        if df[target_col].dtype in [np.int64, np.int32, object]:
            stats["class_distribution"] = df[target_col].value_counts().to_dict()
            stats["n_classes"] = int(df[target_col].nunique())
        else:
            stats["target_mean"]   = float(df[target_col].mean())
            stats["target_std"]    = float(df[target_col].std())
            stats["target_min"]    = float(df[target_col].min())
            stats["target_max"]    = float(df[target_col].max())
    if numeric:
        stats["feature_means"] = {c: round(float(df[c].mean()),4) for c in numeric[:20]}
        stats["feature_stds"]  = {c: round(float(df[c].std()),4)  for c in numeric[:20]}
    return stats


# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler",  StandardScaler())])
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot",  OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(
    experiment_id: int,
    run_id: int,
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    model_name: str,
    params: dict,
    test_size: float = 0.2,
    cv_folds: int = 5,
    tune_hyperparams: bool = False,
    tune_method: str = "grid",
    n_iter: int = 20,
) -> dict:
    """
    Full training pipeline.
    Returns metrics dict.
    """
    t0 = time.time()
    logger.info("Training %s | task=%s | run_id=%d", model_name, task_type, run_id)

    # Separate features and target
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Label encode target for classification
    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))

    # Build pipeline
    preprocessor = build_preprocessor(X)

    if task_type == "classification":
        model_cls = CLASSIFICATION_MODELS[model_name]
    elif task_type == "regression":
        model_cls = REGRESSION_MODELS[model_name]
    else:
        model_cls = CLUSTERING_MODELS[model_name]

    # Merge defaults with user params
    merged_params = {**DEFAULT_PARAMS.get(model_name, {}), **params}
    # Remove params not applicable to this model
    try:
        model_instance = model_cls(**merged_params)
    except TypeError:
        model_instance = model_cls()

    # ── Clustering (no train/test split) ──────────────────────────────────────
    if task_type == "clustering":
        X_proc = preprocessor.fit_transform(X)
        model_instance.fit(X_proc)
        labels = model_instance.labels_ if hasattr(model_instance,"labels_") else model_instance.predict(X_proc)
        sil = float(silhouette_score(X_proc, labels)) if len(set(labels)) > 1 else 0.0
        metrics = {
            "silhouette_score": round(sil, 4),
            "n_clusters":       int(len(set(labels)) - (1 if -1 in labels else 0)),
            "n_samples":        int(len(X)),
        }
        artifact = _save_pipeline(preprocessor, model_instance, run_id, task_type)
        return {"metrics": metrics, "artifact_path": artifact,
                "duration": round(time.time()-t0,2), "feature_importance": {}}

    # ── Supervised ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=(y if task_type=="classification" else None)
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model_instance)])

    # Hyperparameter tuning
    if tune_hyperparams and model_name in PARAM_GRIDS:
        param_grid = PARAM_GRIDS[model_name]
        cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
              if task_type=="classification"
              else KFold(n_splits=cv_folds, shuffle=True, random_state=42))
        scoring = "f1_weighted" if task_type=="classification" else "r2"

        if tune_method == "random":
            search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter,
                                        cv=cv, scoring=scoring,
                                        n_jobs=-1, random_state=42, refit=True)
        else:
            search = GridSearchCV(pipeline, param_grid, cv=cv,
                                  scoring=scoring, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        logger.info("Best params: %s", search.best_params_)
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Cross-validation score
    cv_obj = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
              if task_type=="classification"
              else KFold(n_splits=cv_folds, shuffle=True, random_state=42))
    cv_scoring = "f1_weighted" if task_type=="classification" else "r2"
    cv_scores  = cross_val_score(pipeline, X, y, cv=cv_obj, scoring=cv_scoring, n_jobs=-1)

    # Compute metrics
    if task_type == "classification":
        metrics = {
            "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":        round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall":           round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_score":         round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "cv_mean":          round(float(cv_scores.mean()), 4),
            "cv_std":           round(float(cv_scores.std()), 4),
            "test_size":        int(len(y_test)),
            "train_size":       int(len(y_train)),
        }
        try:
            y_prob = pipeline.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:,1])), 4)
            else:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")), 4)
        except Exception:
            pass
        conf = confusion_matrix(y_test, y_pred).tolist()
        metrics["confusion_matrix"] = conf
    else:
        mse  = float(mean_squared_error(y_test, y_pred))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))
        metrics = {
            "mse":       round(mse, 4),
            "rmse":      round(float(np.sqrt(mse)), 4),
            "mae":       round(mae, 4),
            "r2_score":  round(r2, 4),
            "cv_mean":   round(float(cv_scores.mean()), 4),
            "cv_std":    round(float(cv_scores.std()), 4),
            "test_size": int(len(y_test)),
            "train_size":int(len(y_train)),
        }

    # Feature importance
    feat_imp = _get_feature_importance(pipeline, X.columns.tolist())

    # Save artifacts
    artifact = _save_pipeline(pipeline, None, run_id, task_type)

    duration = round(time.time() - t0, 2)
    logger.info("Training complete: %s in %.1fs", metrics, duration)
    return {
        "metrics":            metrics,
        "artifact_path":      artifact,
        "duration":           duration,
        "feature_importance": feat_imp,
    }


def _get_feature_importance(pipeline: Pipeline, feature_names: list) -> dict:
    try:
        model = pipeline.named_steps.get("model")
        if model is None:
            return {}
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).flatten()
        else:
            return {}
        # Align with preprocessor output
        n = min(len(imp), len(feature_names))
        pairs = sorted(zip(feature_names[:n], imp[:n].tolist()),
                       key=lambda x: x[1], reverse=True)
        return {k: round(v, 4) for k, v in pairs[:20]}
    except Exception:
        return {}


def _save_pipeline(preprocessor, model, run_id: int, task_type: str) -> str:
    path = str(MODELS_DIR / f"run_{run_id}_{task_type}.pkl")
    obj = {"preprocessor": preprocessor, "model": model, "task_type": task_type}
    joblib.dump(obj, path)
    return path


def load_pipeline(artifact_path: str) -> dict:
    return joblib.load(artifact_path)


# ── Drift Detection ───────────────────────────────────────────────────────────
def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                 threshold: float = 0.1) -> dict:
    """
    Statistical drift detection using KS test and PSI.
    """
    from scipy.stats import ks_2samp

    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_drift = {}
    drift_scores  = []

    for col in numeric_cols:
        if col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(ref) < 5 or len(cur) < 5:
            continue

        ks_stat, p_val = ks_2samp(ref, cur)
        psi_score = _compute_psi(ref, cur)

        drifted = ks_stat > threshold or psi_score > 0.2
        feature_drift[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value":       round(float(p_val), 4),
            "psi":           round(float(psi_score), 4),
            "drifted":       drifted,
            "ref_mean":      round(float(ref.mean()), 4),
            "cur_mean":      round(float(cur.mean()), 4),
            "ref_std":       round(float(ref.std()), 4),
            "cur_std":       round(float(cur.std()), 4),
        }
        drift_scores.append(ks_stat)

    overall_score   = float(np.mean(drift_scores)) if drift_scores else 0.0
    drift_detected  = overall_score > threshold
    n_drifted       = sum(1 for v in feature_drift.values() if v.get("drifted"))

    report = {
        "overall_drift_score": round(overall_score, 4),
        "drift_detected":      drift_detected,
        "n_features_checked":  len(feature_drift),
        "n_features_drifted":  n_drifted,
        "drift_pct":           round(n_drifted / max(len(feature_drift),1) * 100, 1),
        "threshold_used":      threshold,
        "method":              "KS-test + PSI",
    }
    return {
        "drift_score":    overall_score,
        "drift_detected": drift_detected,
        "feature_drift":  feature_drift,
        "report":         report,
    }


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    eps = 1e-10
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if max_val == min_val:
        return 0.0
    bins      = np.linspace(min_val, max_val, n_bins+1)
    ref_hist  = np.histogram(reference, bins=bins)[0] / (len(reference) + eps)
    cur_hist  = np.histogram(current, bins=bins)[0] / (len(current) + eps)
    ref_hist  = np.clip(ref_hist, eps, None)
    cur_hist  = np.clip(cur_hist, eps, None)
    psi       = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
    return float(psi)


# ── Synthetic drift data for demo ─────────────────────────────────────────────
def generate_drifted_data(df: pd.DataFrame, drift_magnitude: float = 0.5) -> pd.DataFrame:
    """Generate a version of the dataframe with simulated drift."""
    drifted = df.copy()
    numeric = drifted.select_dtypes(include=[np.number]).columns.tolist()
    rng = np.random.default_rng(42)
    for col in numeric[:5]:
        std = drifted[col].std()
        drifted[col] = drifted[col] + rng.normal(drift_magnitude * std, std * 0.1, size=len(drifted))
    return drifted
