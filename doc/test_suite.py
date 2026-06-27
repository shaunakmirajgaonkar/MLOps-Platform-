"""
tests/test_suite.py — Unit tests for MLOps Platform
Run: pytest tests/ -v
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Use test DB
import app.database as db
db.DB_PATH = Path("/tmp/test_mlops.db")

from app.database import (
    init_db, create_experiment, get_experiments, get_experiment, delete_experiment,
    create_run, update_run, get_runs, get_run,
    register_model, promote_model, get_models, get_model,
    save_dataset, get_datasets,
    save_drift_report, get_drift_reports,
    create_retraining_job, update_retraining_job, get_retraining_jobs,
    log_event, get_logs, get_dashboard_stats,
)
from app.trainer import (
    get_builtin_dataset, compute_dataset_stats,
    train_model, detect_drift, generate_drifted_data,
    _compute_psi, build_preprocessor,
)


@pytest.fixture(autouse=True)
def fresh_db():
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()
    init_db()


# ─────────────────────────────────────────────────────────────────────────────
class TestDatabase:
    def test_init_db(self):
        assert db.DB_PATH.exists()

    def test_create_experiment(self):
        eid = create_experiment("test-exp", "desc", "classification")
        assert eid > 0

    def test_get_experiment(self):
        eid = create_experiment("exp2", "desc", "regression")
        exp = get_experiment(eid)
        assert exp["name"] == "exp2"
        assert exp["task_type"] == "regression"

    def test_duplicate_experiment_fails(self):
        create_experiment("dup", "d", "classification")
        with pytest.raises(Exception):
            create_experiment("dup", "d", "classification")

    def test_delete_experiment(self):
        eid = create_experiment("to-delete", "d", "classification")
        delete_experiment(eid)
        assert get_experiment(eid) is None

    def test_create_run(self):
        eid = create_experiment("run-exp", "d", "classification")
        rid = create_run(eid, "run-1", "Random Forest", {"n_estimators": 100})
        assert rid > 0

    def test_update_run(self):
        eid = create_experiment("run-exp2", "d", "classification")
        rid = create_run(eid, "run-1", "Random Forest", {})
        update_run(rid, "completed", {"accuracy": 0.95}, 12.5, "/path/model.pkl")
        r = get_run(rid)
        assert r["status"] == "completed"
        assert r["metrics"]["accuracy"] == 0.95
        assert r["duration_sec"] == 12.5

    def test_get_runs_for_experiment(self):
        eid = create_experiment("multi-run", "d", "classification")
        create_run(eid, "r1", "Random Forest", {})
        create_run(eid, "r2", "Gradient Boosting", {})
        runs = get_runs(eid)
        assert len(runs) == 2

    def test_register_model(self):
        eid = create_experiment("model-exp", "d", "classification")
        rid = create_run(eid, "r1", "Random Forest", {})
        mid = register_model("test-model", eid, rid,
                              {"accuracy": 0.92}, {}, "/path/m.pkl")
        assert mid > 0

    def test_model_versioning(self):
        eid = create_experiment("ver-exp", "d", "classification")
        rid = create_run(eid, "r1", "Random Forest", {})
        mid1 = register_model("versioned", eid, rid, {}, {}, "/p1")
        mid2 = register_model("versioned", eid, rid, {}, {}, "/p2")
        m1 = get_model(mid1)
        m2 = get_model(mid2)
        assert m1["version"] == 1
        assert m2["version"] == 2

    def test_promote_model(self):
        eid = create_experiment("prom-exp", "d", "classification")
        rid = create_run(eid, "r1", "Random Forest", {})
        mid = register_model("to-promote", eid, rid, {}, {}, "/p")
        promote_model(mid, "production")
        m = get_model(mid)
        assert m["stage"] == "production"

    def test_save_dataset(self):
        did = save_dataset("iris", "v1", "classification", 150, 4,
                           "target", ["f1","f2","f3","f4"], {}, "")
        assert did > 0

    def test_get_datasets(self):
        save_dataset("ds1", "v1", "classification", 100, 5, "y", [], {}, "")
        save_dataset("ds2", "v1", "regression",     200, 10,"y", [], {}, "")
        dbs = get_datasets()
        assert len(dbs) >= 2

    def test_save_drift_report(self):
        drid = save_drift_report(None, None, 0.15, True,
                                  {"f1":{"ks_statistic":0.2}}, {"overall":0.15})
        assert drid > 0

    def test_get_drift_reports(self):
        save_drift_report(None, None, 0.05, False, {}, {})
        reps = get_drift_reports()
        assert len(reps) >= 1

    def test_retraining_job(self):
        eid = create_experiment("rt-exp", "d", "classification")
        rid = create_run(eid, "r1", "Random Forest", {})
        mid = register_model("rt-model", eid, rid, {}, {}, "/p")
        jid = create_retraining_job(mid, "drift detected")
        assert jid > 0
        update_retraining_job(jid, "running")
        update_retraining_job(jid, "completed", rid)
        jobs = get_retraining_jobs()
        assert any(j["id"]==jid for j in jobs)

    def test_log_event(self):
        log_event("INFO", "Test", "Test message", {"key": "value"})
        logs = get_logs(10)
        assert any(l["message"]=="Test message" for l in logs)

    def test_dashboard_stats(self):
        s = get_dashboard_stats()
        assert "total_experiments" in s
        assert "total_models" in s
        assert "drift_alerts" in s


# ─────────────────────────────────────────────────────────────────────────────
class TestDatasets:
    def test_load_iris(self):
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        assert tc == "target"
        assert tt == "classification"
        assert len(df) == 150

    def test_load_wine(self):
        df, tc, tt = get_builtin_dataset("Wine Quality (Classification)")
        assert tt == "classification"
        assert len(df) > 0

    def test_load_breast_cancer(self):
        df, tc, tt = get_builtin_dataset("Breast Cancer (Classification)")
        assert tt == "classification"
        assert len(df) == 569

    def test_load_diabetes_regression(self):
        df, tc, tt = get_builtin_dataset("Diabetes (Regression)")
        assert tt == "regression"
        assert len(df) == 442

    def test_load_synthetic_classification(self):
        df, tc, tt = get_builtin_dataset("Synthetic Classification")
        assert tt == "classification"
        assert len(df) == 1000

    def test_load_synthetic_regression(self):
        df, tc, tt = get_builtin_dataset("Synthetic Regression")
        assert tt == "regression"

    def test_load_synthetic_clustering(self):
        df, tc, tt = get_builtin_dataset("Synthetic Clustering")
        assert tt == "clustering"

    def test_unknown_dataset(self):
        with pytest.raises(ValueError):
            get_builtin_dataset("Unknown Dataset XYZ")

    def test_compute_stats_classification(self):
        df, tc, _ = get_builtin_dataset("Iris (Classification)")
        stats = compute_dataset_stats(df, tc)
        assert "shape" in stats
        assert "n_classes" in stats
        assert stats["n_classes"] == 3

    def test_compute_stats_regression(self):
        df, tc, _ = get_builtin_dataset("Diabetes (Regression)")
        stats = compute_dataset_stats(df, tc)
        assert "target_mean" in stats


# ─────────────────────────────────────────────────────────────────────────────
class TestTrainer:
    def setup_method(self):
        self.eid = create_experiment("train-test", "desc", "classification")

    def test_random_forest_classification(self):
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        rid = create_run(self.eid, "rf-test", "Random Forest", {})
        result = train_model(self.eid, rid, df, tc, tt, "Random Forest", {})
        assert "metrics" in result
        assert result["metrics"]["accuracy"] > 0.7
        assert result["duration"] > 0

    def test_logistic_regression(self):
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        rid = create_run(self.eid, "lr-test", "Logistic Regression", {})
        result = train_model(self.eid, rid, df, tc, tt, "Logistic Regression", {})
        assert result["metrics"]["accuracy"] > 0.7

    def test_gradient_boosting(self):
        df, tc, tt = get_builtin_dataset("Breast Cancer (Classification)")
        rid = create_run(self.eid, "gb-test", "Gradient Boosting", {})
        result = train_model(self.eid, rid, df, tc, tt, "Gradient Boosting",
                             {"n_estimators":50})
        assert result["metrics"]["accuracy"] > 0.85

    def test_regression_model(self):
        exp = create_experiment("reg-test", "d", "regression")
        df, tc, tt = get_builtin_dataset("Diabetes (Regression)")
        rid = create_run(exp, "ridge-test", "Ridge Regression", {})
        result = train_model(exp, rid, df, tc, tt, "Ridge Regression", {})
        assert "r2_score" in result["metrics"]
        assert result["metrics"]["rmse"] > 0

    def test_clustering_model(self):
        exp = create_experiment("clust-test", "d", "clustering")
        df, tc, tt = get_builtin_dataset("Synthetic Clustering")
        rid = create_run(exp, "km-test", "K-Means", {})
        result = train_model(exp, rid, df, tc, tt, "K-Means",
                             {"n_clusters":4,"n_init":5})
        assert "silhouette_score" in result["metrics"]

    def test_metrics_saved_to_db(self):
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        rid = create_run(self.eid, "metrics-test", "Random Forest", {})
        result = train_model(self.eid, rid, df, tc, tt, "Random Forest", {})
        update_run(rid, "completed", result["metrics"], result["duration"])
        r = get_run(rid)
        assert r["metrics"]["accuracy"] > 0

    def test_feature_importance_returned(self):
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        rid = create_run(self.eid, "fi-test", "Random Forest", {})
        result = train_model(self.eid, rid, df, tc, tt, "Random Forest", {})
        assert isinstance(result["feature_importance"], dict)
        assert len(result["feature_importance"]) > 0

    def test_confusion_matrix_in_metrics(self):
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        rid = create_run(self.eid, "cm-test", "Random Forest", {})
        result = train_model(self.eid, rid, df, tc, tt, "Random Forest", {})
        assert "confusion_matrix" in result["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
class TestDriftDetection:
    def test_no_drift_same_data(self):
        df, _, _ = get_builtin_dataset("Iris (Classification)")
        result = detect_drift(df, df.copy(), threshold=0.1)
        assert result["drift_score"] < 0.01
        assert result["drift_detected"] is False

    def test_drift_detected_with_shift(self):
        df, _, _ = get_builtin_dataset("Synthetic Classification")
        df_drifted = generate_drifted_data(df, drift_magnitude=2.0)
        result = detect_drift(df, df_drifted, threshold=0.1)
        assert result["drift_detected"] is True
        assert result["drift_score"] > 0.1

    def test_feature_drift_returned(self):
        df, _, _ = get_builtin_dataset("Iris (Classification)")
        df_d = generate_drifted_data(df, 1.0)
        result = detect_drift(df, df_d)
        assert isinstance(result["feature_drift"], dict)
        assert len(result["feature_drift"]) > 0

    def test_feature_drift_fields(self):
        df, _, _ = get_builtin_dataset("Iris (Classification)")
        df_d = generate_drifted_data(df, 1.0)
        result = detect_drift(df, df_d)
        for feat, vals in result["feature_drift"].items():
            assert "ks_statistic" in vals
            assert "p_value" in vals
            assert "psi" in vals
            assert "drifted" in vals

    def test_psi_same_distribution(self):
        data = np.random.normal(0, 1, 1000)
        psi  = _compute_psi(data, data.copy())
        assert psi < 0.05

    def test_psi_different_distribution(self):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)
        psi = _compute_psi(ref, cur)
        assert psi > 1.0

    def test_generate_drifted_data(self):
        df, _, _ = get_builtin_dataset("Iris (Classification)")
        df_d = generate_drifted_data(df, 1.0)
        assert df_d.shape == df.shape
        numeric = df.select_dtypes(include=[np.number]).columns
        for col in numeric[:3]:
            assert df[col].mean() != df_d[col].mean()

    def test_drift_report_saved(self):
        df, _, _ = get_builtin_dataset("Iris (Classification)")
        df_d   = generate_drifted_data(df, 1.5)
        result = detect_drift(df, df_d)
        drid   = save_drift_report(None, None,
                                   result["drift_score"],
                                   result["drift_detected"],
                                   result["feature_drift"],
                                   result["report"])
        assert drid > 0
        reports = get_drift_reports()
        assert any(r["id"]==drid for r in reports)


# ─────────────────────────────────────────────────────────────────────────────
class TestEndToEnd:
    def test_full_mlops_pipeline(self):
        """Simulate: create experiment → train → register → promote → detect drift → retrain."""
        # 1. Create experiment
        eid = create_experiment("e2e-test", "End to end test", "classification")
        assert eid > 0

        # 2. Load dataset
        df, tc, tt = get_builtin_dataset("Iris (Classification)")
        did = save_dataset("iris-e2e", "v1", tt, len(df), len(df.columns)-1,
                           tc, list(df.columns[:-1]), {}, "")
        assert did > 0

        # 3. Train model
        rid = create_run(eid, "e2e-run", "Random Forest", {"n_estimators":50})
        result = train_model(eid, rid, df, tc, tt, "Random Forest", {"n_estimators":50})
        update_run(rid, "completed", result["metrics"], result["duration"], result["artifact_path"])
        assert result["metrics"]["accuracy"] > 0.8

        # 4. Register model
        mid = register_model("e2e-model", eid, rid,
                              result["metrics"], {"n_estimators":50},
                              result["artifact_path"], "E2E test model")
        m   = get_model(mid)
        assert m["stage"] == "staging"

        # 5. Promote to production
        promote_model(mid, "production")
        m = get_model(mid)
        assert m["stage"] == "production"

        # 6. Detect drift
        df_d    = generate_drifted_data(df, 1.5)
        drift_r = detect_drift(df, df_d)
        drid    = save_drift_report(mid, did, drift_r["drift_score"],
                                    drift_r["drift_detected"],
                                    drift_r["feature_drift"], drift_r["report"])
        assert drid > 0

        # 7. Trigger retraining
        jid = create_retraining_job(mid, "drift detected")
        update_retraining_job(jid, "running")
        rid2    = create_run(eid, "e2e-retrain", "Random Forest", {"n_estimators":50})
        result2 = train_model(eid, rid2, df, tc, tt, "Random Forest", {"n_estimators":50})
        update_run(rid2, "completed", result2["metrics"], result2["duration"])
        mid2    = register_model("e2e-model", eid, rid2, result2["metrics"],
                                  {}, result2["artifact_path"], "Retrained")
        update_retraining_job(jid, "completed", rid2)

        # 8. Verify stats
        stats = get_dashboard_stats()
        assert stats["total_experiments"] >= 1
        assert stats["total_runs"] >= 2
        assert stats["total_models"] >= 2

        jobs = get_retraining_jobs()
        assert any(j["id"]==jid and j["status"]=="completed" for j in jobs)
