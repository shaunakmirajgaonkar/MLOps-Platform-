# MLOps Concepts — Simple Explanation

## What is MLOps?

MLOps = Machine Learning + Operations

Just like DevOps keeps software running in production, MLOps keeps ML models running accurately in production.

---

## Key Concepts

### Experiment
A project container. Groups all training runs for one problem.
Example: "fraud-detection", "heart-disease-prediction"

### Run
One training attempt. Different algorithm or parameters = different run.
Example: Run 1 = Random Forest 94%, Run 2 = Gradient Boosting 96%

### Model Registry
A versioned store of trained models.
Example: fraud-detector v1 (staging) → v2 (production) → v3 (champion)

### Data Drift
When real-world data changes from what the model was trained on.
Example: Model trained on 2023 data, deployed in 2024 when patterns changed.

### KS-Test
Kolmogorov-Smirnov test. Compares two distributions statistically.
Returns a score 0-1. Higher = more different.

### PSI
Population Stability Index. Measures how much a distribution has shifted.
PSI < 0.1 = stable, PSI 0.1-0.2 = monitor, PSI > 0.2 = retrain

### Auto Retraining
Automatically training a new model when drift is detected.
No manual work needed.

---

## Why This Matters

Without MLOps:
- You retrain manually when someone complains model is wrong
- No record of which model version is running
- No way to roll back to a previous version
- No early warning when data changes

With MLOps:
- Drift detected automatically before accuracy drops
- Full version history of every model
- One-click rollback to any previous version
- Audit trail of every action taken
