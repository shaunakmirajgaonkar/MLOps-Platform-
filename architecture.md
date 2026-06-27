# Architecture — MLOps Platform

## Overview

```
Browser (localhost:8501)
         │
         ▼
    app.py (Streamlit UI — 9 pages)
         │
    ┌────┴──-------- ──┐
    │                  │
app/database.py    app/trainer.py
(SQLite — 7 tables) (ML engine + drift)
                         │
                    Ollama / scikit-learn
                    Local model artifacts
                    (models/*.pkl)
```

## Data Flow

```
User selects dataset + algorithm
        ↓
trainer.py builds preprocessing pipeline
        ↓
Training runs with cross-validation
        ↓
Metrics computed and returned
        ↓
database.py saves run + metrics to SQLite
        ↓
Model artifact saved to models/ via joblib
        ↓
Results rendered in Streamlit UI
```

## Database Schema (7 tables)

| Table | Purpose |
|---|---|
| experiments | Project containers |
| runs | Individual training attempts |
| models | Registered model versions |
| datasets | Dataset metadata |
| drift_reports | Drift detection results |
| retraining_jobs | Auto retraining history |
| pipeline_logs | Audit trail |

## Drift Detection Flow

```
Reference dataset (training data)
        ↓
Current dataset (new production data)
        ↓
KS-test per feature (scipy.stats.ks_2samp)
        ↓
PSI per feature (custom implementation)
        ↓
Overall drift score = mean(KS statistics)
        ↓
If score > threshold → drift_detected = True
        ↓
Alert raised → Auto retraining triggered
```
