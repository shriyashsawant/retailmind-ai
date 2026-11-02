# RetailMind AI

> **One retail AI platform, built one module per day.**
> RetailMind turns raw retail data into decisions a business can act on — starting with customer churn.

---

## Product vision

RetailMind AI is an incremental, end-to-end retail intelligence platform. Each day adds one production-shaped module to the same codebase: a data pipeline, a model, a service, an API, a dashboard. By Day 35 the result is a single deployed product, not a folder of disconnected notebooks.

**Day 1 goal:** give the business an immediate, trustworthy churn baseline — flag customers likely to leave, with a model that can be called as a service.

---

## What ships in this increment

- A clean, persistent repo structure (`data/`, `src/`, `models/`, `notebooks/`, `reports/`)
- A reproducible **ingest → clean → split → preprocess → train → evaluate → persist** pipeline
- A fitted **scikit-learn Pipeline** persisted with `joblib`
- A thin **ML service** (`predict()`) plus a CLI for scoring one or many customers
- A recorded baseline that every future increment must beat

---

## Dataset

**Telco Customer Churn** — [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7,043 customers · 21 columns · binary target `Churn` (`Yes` / `No`)

A copy lives at `data/raw/Telco-Customer-Churn.csv` so the repo runs out of the box. If you re-download from Kaggle, keep the same filename and location.

Key data facts:

| Check | Result |
| --- | --- |
| Rows / columns | 7,043 / 21 |
| Churn rate | **26.54%** (1,869 churned) |
| Missing values | `TotalCharges` has **11 blank strings** |
| Identifier | `customerID` (dropped from features) |

`TotalCharges` arrives as text with 11 blank values — all new customers with `tenure = 0`. They are coerced to numeric and filled with `0.0`, which is the correct business value for a customer who has not been billed yet.

---

## Repository structure

```text
retailmind-ai/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── src/
│   ├── config.py        # paths, constants, seed
│   ├── data.py          # load, clean, profile
│   ├── pipeline.py      # preprocessing + baseline estimator
│   ├── evaluate.py      # metrics and confusion matrix
│   ├── train.py         # end-to-end training entry point
│   └── service.py       # predict() helper + CLI
├── models/              # persisted pipeline + metrics (generated)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
├── reports/             # generated data profile
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train, evaluate and persist the baseline
python -m src.train

# Score the built-in demo customers
python -m src.service

# Or score a CSV of customers
python -m src.service --input path/to/customers.csv
```

Outputs:

- `models/churn_pipeline.joblib` — the fitted end-to-end pipeline
- `models/metrics.json` — baseline metrics for later comparison
- `reports/data_profile.json` — schema, missing values and target balance

---

## How the pipeline works

```text
CSV
 └─ clean TotalCharges → map Churn to 0/1 → drop customerID
     └─ stratified 80/20 train-test split (random_state=42)
         └─ ColumnTransformer
             ├─ StandardScaler        → SeniorCitizen, tenure, MonthlyCharges, TotalCharges
             └─ OneHotEncoder         → all categorical columns (handle_unknown="ignore")
                 └─ LogisticRegression(max_iter=2000)
                     └─ evaluate: accuracy, precision, recall, F1, ROC-AUC
                         └─ joblib.dump(pipeline + metadata)
```

Everything up to the classifier lives inside one `Pipeline`, so serving cannot drift from training: the exact same encoders and scaler are applied to every prediction.

---

## Day 1 baseline results

Hold-out set: **1,409 customers** (374 churned). Decision threshold: **0.50**.

| Metric | Score |
| --- | ---: |
| Accuracy | **0.8055** |
| Precision (churn) | **0.6572** |
| Recall (churn) | **0.5588** |
| F1 (churn) | **0.6040** |
| ROC-AUC | **0.8421** |

Confusion matrix:

|  | Predicted stay | Predicted churn |
| --- | ---: | ---: |
| **Actually stayed** | 926 | 109 |
| **Actually churned** | 165 | 209 |

Reading it: the model catches about **56% of churners** (recall) and is right about **66% of the time** when it flags someone (precision). ROC-AUC of **0.84** means the ranking signal is strong; the next increments should convert that ranking into better recall.

> These numbers are the benchmark. Every later day must be measured against **ROC-AUC 0.8421** and **F1 0.6040**.

---

## Using the service

```python
from src.service import predict

customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}

predict(customer)
# [{'churn_probability': 0.6137, 'churn_prediction': 'Yes', 'risk_tier': 'medium'}]
```

`predict()` accepts a single dict, a list of dicts, or a DataFrame, and returns one result per customer. Risk tiers are a presentation convenience: `high` ≥ 0.70, `medium` ≥ 0.40, otherwise `low`.

---

## Design notes

- **Baseline first.** Logistic regression is deliberately simple: fast, interpretable, and a fair reference point for the stronger models coming next.
- **No leakage.** Cleaning and splitting happen before fitting; all encoding/scaling is learned inside the pipeline on training data only.
- **Reproducible.** Fixed `random_state=42` and a stratified split preserve the churn rate in both folds.
- **Persist the whole pipeline**, not just the estimator — preprocessing travels with the model.
- **Class imbalance is visible.** At 26.5% churn, accuracy alone flatters the model, so precision, recall and ROC-AUC are reported alongside it.

---

## Roadmap

- [x] **Day 1** — End-to-end churn pipeline + persisted ML service + baseline metrics
- [ ] **Day 2** — Swap the baseline for Decision Tree, Random Forest and SVM; pick a champion
- [ ] Later — feature engineering, class-imbalance handling, API layer, monitoring, deployment

---

## Tech stack

`pandas` · `scikit-learn` · `joblib` — Python 3.11+

Built as part of the **RetailMind AI** 35-day incremental project.
