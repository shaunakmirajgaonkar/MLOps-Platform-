# MLOps Platform 🤖

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?style=flat-square)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue?style=flat-square)
![Plotly](https://img.shields.io/badge/Charts-Plotly-purple?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-45%20passing-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> End-to-end MLOps platform for managing the complete machine learning lifecycle — experiment tracking, multi-model training, model registry, drift detection, and automated retraining.

---

## What is MLOps Platform?

MLOps Platform is a **production-style machine learning operations system** that automates the full ML lifecycle. Instead of running Python scripts manually, it provides a web interface where you can train models, monitor them, detect data drift, and retrain automatically — all with one click.

```
Upload Data → Train Model → Register → Monitor → Detect Drift → Auto Retrain
```

---

## Demo

```
Create Experiment → Load Dataset → Train Model → Register → Promote to Production → Detect Drift → Auto Retrain
```

**Built-in datasets:**
```
Iris                    → Classification (150 samples)
Wine Quality            → Classification (178 samples)
Breast Cancer           → Classification (569 samples)
Diabetes                → Regression     (442 samples)
Synthetic Classification→ Classification (1,000 samples)
Synthetic Regression    → Regression     (1,000 samples)
Synthetic Clustering    → Clustering     (500 samples)
```

---

## Features

| Feature | Description |
|---|---|
| 🧪 **Experiment Tracking** | Create experiments, log runs, compare results |
| 🚀 **Multi-Model Training** | Random Forest, Gradient Boosting, Logistic Regression, SVM, Ridge, Lasso, K-Means, DBSCAN |
| 🔧 **Hyperparameter Tuning** | Grid Search and Randomized Search with cross-validation |
| 📦 **Model Registry** | Version control, staging/production/champion promotion |
| 📊 **Data Explorer** | Feature distributions, correlation matrix, dataset stats |
| 🔍 **Drift Detection** | KS-test + Population Stability Index (PSI) per feature |
| 🔄 **Auto Retraining** | Triggered automatically on drift detection |
| 📈 **Monitoring Dashboard** | Run timeline, metric trends, model health charts |
| 📋 **Pipeline Logs** | Full structured audit log with download |
| ✅ **45 Unit Tests** | Full test coverage across all modules |

---

## Quickstart

```bash
git clone https://github.com/shaunakmirajgaonkar/mlops-platform.git
cd mlops-platform
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**

---

## How It Works

```
You select a dataset and algorithm
           ↓
Training Agent preprocesses, trains, evaluates, saves model
           ↓
Model registered in registry with version number
           ↓
Promoted to production
           ↓
Drift Detection Agent monitors feature distributions (KS-test + PSI)
           ↓
If drift detected → Auto Retraining Agent retrains automatically
           ↓
New model version registered — zero manual work
```

---

## ML Models Supported

**Classification**
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)

**Regression**
- Random Forest
- Gradient Boosting
- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)

**Clustering**
- K-Means
- DBSCAN

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Plotly |
| ML Engine | scikit-learn |
| Drift Detection | SciPy (KS-test) + custom PSI |
| Database | SQLite (WAL mode) |
| Model Storage | joblib |
| Testing | pytest (45 tests) |
| CI/CD | GitHub Actions |

---

## Project Structure

```
mlops-platform/
├── app.py                      ← Main Streamlit application (9 pages)
├── app/
│   ├── __init__.py
│   ├── database.py             ← SQLite layer (7 tables)
│   └── trainer.py              ← ML engine + drift detection
├── tests/
│   └── test_suite.py           ← 45 unit + integration tests
├── data/                       ← SQLite database (auto-created)
├── models/                     ← Saved model artifacts
├── .github/workflows/ci.yml    ← GitHub Actions CI
├── docs/
│   ├── architecture.md
│   ├── mlops_concepts.md
│   └── supported_models.md
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── ACKNOWLEDGMENTS.md
├── Run Instructions
├── .env.example
└── .gitignore
```

---

## Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | KPI tiles, run charts, live activity log |
| 🧪 Experiments | Create/manage experiments, view run history |
| 🚀 Train Model | Configure and train with real-time progress |
| 📦 Model Registry | Version control with stage promotion |
| 📊 Data Management | Dataset explorer with correlation matrix |
| 🔍 Drift Detection | KS-test + PSI with feature-level charts |
| 🔄 Auto Retraining | Trigger and monitor retraining jobs |
| 📈 Monitoring | Performance trends and model health |
| 📋 Pipeline Logs | Structured audit log with filtering |

---

## Tests

```bash
pytest tests/test_suite.py -v
```

```
45 passed in ~7s
```

Covers: database CRUD, dataset loading, all model types, drift detection, PSI computation, full end-to-end pipeline.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Shaunak Mirajgaonkar**
BE Computer Engineering — MMCOE Pune (SPPU)
[GitHub](https://github.com/shaunakmirajgaonkar)
