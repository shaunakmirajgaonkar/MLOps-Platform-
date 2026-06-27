MLOps Platform — Run Instructions
===================================

PREREQUISITES
-------------
- Python 3.10 or higher
- At least 4 GB RAM


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


STEP 4 — Run tests (optional)
-------------------------------
  pytest tests/test_suite.py -v
  Expected: 45 passed


STEP 5 — Launch the platform
------------------------------
  streamlit run app.py

  Open browser at: http://localhost:8501


EVERY TIME YOU WANT TO RUN
----------------------------
  cd mlops-platform
  source .venv/bin/activate
  streamlit run app.py


STOPPING THE APP
-----------------
  Ctrl+C in terminal
