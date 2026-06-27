MLOps Platform — Run Instructions
===================================

PREREQUISITES
-------------
• Python 3.10 or higher
• At least 4 GB RAM


STEP 1 — Clone the repository
------------------------------
  git clone https://github.com/shaunakmirajgaonkar/mlops-platform.git
  cd mlops-platform


STEP 2 — Create virtual environment
-------------------------------------
  python3 -m venv .venv
  source .venv/bin/activate        (Mac/Linux)
  .venv\Scripts\activate           (Windows)


STEP 3 — Install dependencies
-------------------------------
  pip install -r requirements.txt


STEP 4 — Run tests (optional but recommended)
----------------------------------------------
  pytest tests/test_suite.py -v
  Expected: 45 passed


STEP 5 — Launch the platform
------------------------------
  streamlit run app.py

  Open browser at: http://localhost:8501


USAGE — QUICK DEMO
-------------------
1. Click 🧪 Experiments → Create new experiment
2. Click 🚀 Train Model → Load dataset → Select algorithm → Train
3. Click 📦 Model Registry → Promote model to production
4. Click 🔍 Drift Detection → Run drift check
5. Click 🔄 Auto Retraining → Trigger retraining
6. Click 📈 Monitoring → View all charts
7. Click 📋 Pipeline Logs → View audit trail


STOPPING THE APP
-----------------
  Ctrl+C in terminal


EVERY TIME YOU WANT TO RUN
----------------------------
  cd mlops-platform
  source .venv/bin/activate
  streamlit run app.py
