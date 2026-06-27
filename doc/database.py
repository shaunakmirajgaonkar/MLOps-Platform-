"""app/database.py — SQLite persistence for MLOps Platform"""
import sqlite3, json, logging
import numpy as np

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _dumps(obj): return json.dumps(obj, cls=_SafeEncoder)
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path("data/mlops.db")
logger  = logging.getLogger(__name__)

def _now(): return datetime.now(timezone.utc).isoformat()

@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn; conn.commit()
    except Exception:
        conn.rollback(); raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL, description TEXT,
                task_type TEXT NOT NULL, status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                run_name TEXT NOT NULL, model_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                params TEXT DEFAULT '{}', metrics TEXT DEFAULT '{}',
                tags TEXT DEFAULT '{}', artifact_path TEXT,
                duration_sec REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, version INTEGER NOT NULL,
                experiment_id INTEGER, run_id INTEGER,
                stage TEXT DEFAULT 'staging',
                metrics TEXT DEFAULT '{}', params TEXT DEFAULT '{}',
                artifact_path TEXT, description TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(name, version)
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, version TEXT NOT NULL,
                task_type TEXT NOT NULL, num_samples INTEGER,
                num_features INTEGER, target_column TEXT,
                feature_names TEXT DEFAULT '[]', stats TEXT DEFAULT '{}',
                artifact_path TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER, dataset_id INTEGER,
                drift_score REAL, drift_detected INTEGER DEFAULT 0,
                feature_drift TEXT DEFAULT '{}', report TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retraining_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER, trigger TEXT NOT NULL,
                status TEXT DEFAULT 'pending', new_run_id INTEGER,
                started_at TEXT, completed_at TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pipeline_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL, component TEXT NOT NULL,
                message TEXT NOT NULL, details TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
        """)

def _parse(rows):
    result = []
    for r in rows:
        d = dict(r)
        for k in ["params","metrics","tags","stats","feature_drift","report","feature_names","details"]:
            if k in d:
                try: d[k] = json.loads(d[k] or "{}")
                except: pass
        result.append(d)
    return result

# ── Experiments ──────────────────────────────────────────────────────────────
def create_experiment(name, description, task_type):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO experiments (name,description,task_type,created_at,updated_at) VALUES(?,?,?,?,?)",
            (name, description, task_type, _now(), _now()))
        return cur.lastrowid

def get_experiments():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT e.*,COUNT(r.id) as run_count FROM experiments e
            LEFT JOIN runs r ON r.experiment_id=e.id
            GROUP BY e.id ORDER BY e.created_at DESC""").fetchall()
    return [dict(r) for r in rows]

def get_experiment(exp_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id=?",(exp_id,)).fetchone()
    return dict(row) if row else None

def delete_experiment(exp_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM runs WHERE experiment_id=?",(exp_id,))
        conn.execute("DELETE FROM experiments WHERE id=?",(exp_id,))

# ── Runs ──────────────────────────────────────────────────────────────────────
def create_run(experiment_id, run_name, model_type, params):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (experiment_id,run_name,model_type,status,params,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (experiment_id, run_name, model_type, "running", json.dumps(params), _now(), _now()))
        return cur.lastrowid

def update_run(run_id, status, metrics, duration_sec, artifact_path=""):
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET status=?,metrics=?,duration_sec=?,artifact_path=?,updated_at=? WHERE id=?",
            (status, json.dumps(metrics), duration_sec, artifact_path, _now(), run_id))

def get_runs(experiment_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE experiment_id=? ORDER BY created_at DESC",(experiment_id,)).fetchall()
    return _parse(rows)

def get_all_runs():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT r.*,e.name as experiment_name FROM runs r
            JOIN experiments e ON r.experiment_id=e.id
            ORDER BY r.created_at DESC LIMIT 500""").fetchall()
    return _parse(rows)

def get_run(run_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone()
    return _parse([row])[0] if row else None

# ── Models ────────────────────────────────────────────────────────────────────
def register_model(name, experiment_id, run_id, metrics, params, artifact_path, description=""):
    with get_conn() as conn:
        v = (conn.execute("SELECT COALESCE(MAX(version),0) FROM models WHERE name=?",(name,)).fetchone()[0] or 0) + 1
        cur = conn.execute(
            "INSERT INTO models (name,version,experiment_id,run_id,stage,metrics,params,artifact_path,description,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (name,v,experiment_id,run_id,"staging",json.dumps(metrics),json.dumps(params),artifact_path,description,_now(),_now()))
        return cur.lastrowid

def promote_model(model_id, stage):
    with get_conn() as conn:
        conn.execute("UPDATE models SET stage=?,updated_at=? WHERE id=?",(stage,_now(),model_id))

def get_models(stage=None):
    with get_conn() as conn:
        if stage:
            rows = conn.execute("SELECT * FROM models WHERE stage=? ORDER BY created_at DESC",(stage,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    return _parse(rows)

def get_model(model_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM models WHERE id=?",(model_id,)).fetchone()
    return _parse([row])[0] if row else None

# ── Datasets ──────────────────────────────────────────────────────────────────
def save_dataset(name, version, task_type, num_samples, num_features, target_column, feature_names, stats, artifact_path):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO datasets (name,version,task_type,num_samples,num_features,target_column,feature_names,stats,artifact_path,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (name,version,task_type,num_samples,num_features,target_column,json.dumps(feature_names),json.dumps(stats),artifact_path,_now()))
        return cur.lastrowid

def get_datasets():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    return _parse(rows)

# ── Drift ──────────────────────────────────────────────────────────────────────
def save_drift_report(model_id, dataset_id, drift_score, drift_detected, feature_drift, report):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO drift_reports (model_id,dataset_id,drift_score,drift_detected,feature_drift,report,created_at) VALUES(?,?,?,?,?,?,?)",
            (model_id,dataset_id,drift_score,int(drift_detected),_dumps(feature_drift),_dumps(report),_now()))
        return cur.lastrowid

def get_drift_reports(model_id=None):
    with get_conn() as conn:
        if model_id:
            rows = conn.execute("SELECT * FROM drift_reports WHERE model_id=? ORDER BY created_at DESC",(model_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT 50").fetchall()
    return _parse(rows)

# ── Retraining ────────────────────────────────────────────────────────────────
def create_retraining_job(model_id, trigger):
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO retraining_jobs (model_id,trigger,status,created_at) VALUES(?,?,?,?)",
                           (model_id,trigger,"pending",_now()))
        return cur.lastrowid

def update_retraining_job(job_id, status, new_run_id=None):
    ts = _now()
    with get_conn() as conn:
        if status == "running":
            conn.execute("UPDATE retraining_jobs SET status=?,started_at=? WHERE id=?",(status,ts,job_id))
        else:
            conn.execute("UPDATE retraining_jobs SET status=?,completed_at=?,new_run_id=? WHERE id=?",(status,ts,new_run_id,job_id))

def get_retraining_jobs():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM retraining_jobs ORDER BY created_at DESC LIMIT 50").fetchall()
    return [dict(r) for r in rows]

# ── Logs ──────────────────────────────────────────────────────────────────────
def log_event(level, component, message, details=None):
    with get_conn() as conn:
        conn.execute("INSERT INTO pipeline_logs (level,component,message,details,created_at) VALUES(?,?,?,?,?)",
                     (level,component,message,json.dumps(details or {}),_now()))

def get_logs(limit=100):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM pipeline_logs ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
    return _parse(rows)

# ── Stats ─────────────────────────────────────────────────────────────────────
def get_dashboard_stats():
    with get_conn() as conn:
        return {
            "total_experiments": conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
            "total_runs":        conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "successful_runs":   conn.execute("SELECT COUNT(*) FROM runs WHERE status='completed'").fetchone()[0],
            "failed_runs":       conn.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0],
            "total_models":      conn.execute("SELECT COUNT(*) FROM models").fetchone()[0],
            "production_models": conn.execute("SELECT COUNT(*) FROM models WHERE stage='production'").fetchone()[0],
            "total_datasets":    conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "drift_alerts":      conn.execute("SELECT COUNT(*) FROM drift_reports WHERE drift_detected=1").fetchone()[0],
            "pending_retrain":   conn.execute("SELECT COUNT(*) FROM retraining_jobs WHERE status='pending'").fetchone()[0],
        }
