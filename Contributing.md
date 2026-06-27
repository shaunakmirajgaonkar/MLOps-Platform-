# Contributing to MLOps Platform

Thank you for considering contributing! All contributions are welcome.

---

## Getting Started

```bash
git clone https://github.com/shaunakmirajgaonkar/mlops-platform.git
cd mlops-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Make Changes and Test

```bash
pytest tests/test_suite.py -v
streamlit run app.py
```

## Commit and Push

```bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

Then open a Pull Request.

---

## Contribution Areas

- [ ] Add new ML models (XGBoost, LightGBM)
- [ ] Add SHAP explainability
- [ ] Add email alerts on drift detection
- [ ] Add model comparison page
- [ ] Add PostgreSQL support
- [ ] Add Docker support

---

## Code Style

- Follow PEP 8
- Add docstrings to all functions
- Write tests for new features
- Keep functions short and single-purpose

---

## Reporting Bugs

Open an issue with:
- OS and Python version
- Steps to reproduce
- Error message or screenshot
- Expected vs actual behaviour
