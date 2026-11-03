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
│   ├── config.py        # file names, column names, settings
│   ├── data.py          # load, clean, split the data
│   ├── pipeline.py      # data preparation + model, joined together
│   ├── evaluate.py      # scores (recall, precision, F1 ...)
│   ├── train.py         # Day 1: train the logistic regression baseline
│   ├── train_day2.py    # Day 2: compare Decision Tree, Random Forest, SVM
│   └── service.py       # predict() helper + command line
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

# Day 1: train, test and save the baseline model
python -m src.train

# Day 2: train Decision Tree, Random Forest and SVM, compare, save the champion
python -m src.train_day2

# Score the built-in demo customers (uses the Day 2 champion)
python -m src.service

# Or score a CSV of customers
python -m src.service path/to/customers.csv
```

Run every command from the main project folder (the one that contains `src`).

Outputs:

- `models/churn_pipeline.joblib` — the Day 1 model (logistic regression)
- `models/metrics.json` — Day 1 scores for later comparison
- `models/champion_pipeline.joblib` — the Day 2 champion (best of the three new models)
- `models/day2_comparison.json` — every model's scores and why the champion won
- `reports/data_profile.json` — a few simple facts about the raw data

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
                         └─ joblib.dump(pipeline)
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

## Day 2 — Decision Tree vs Random Forest vs SVM

Goal: catch more of the customers who are about to leave. All three new models use the **same Day 1 data preparation** (`make_pipeline`), the **same train/test customers** and the **same random seed (42)**, so the comparison is fair. Each one uses `class_weight="balanced"`, which tells the model that a churner matters more than a stayer (only about 1 in 4 customers churns).

Hold-out set: **1,409 customers** (374 churned). Decision threshold: **0.50**. The champion is the model with the highest recall on the churn class.

| Model | Recall (churn) | Precision (churn) | F1 | ROC-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (Day 1) | 0.5588 | **0.6572** | 0.6040 | **0.8421** | **0.8055** |
| Decision Tree (max depth 5) | 0.7567 | 0.5260 | 0.6206 | 0.8318 | 0.7544 |
| **Random Forest (200 trees, max depth 8)** | **0.7914** | 0.5258 | **0.6318** | 0.8407 | 0.7551 |
| SVM (RBF) | 0.6257 | 0.5894 | 0.6070 | 0.8210 | 0.7850 |

**Champion: Random Forest.** It finds **79% of churners** (Day 1 found 56%), so the retention team reaches far more at-risk customers, and its F1 is the best of all four. The price is lower precision (0.53 vs 0.66): about half of the customers it flags will not actually leave, which is acceptable when a retention offer is cheap compared with losing a customer. Its ROC-AUC (0.8407) is about the same as Day 1 (0.8421), which tells us the gain comes from catching more churners, not from a better ranking of customers. The champion is saved to `models/champion_pipeline.joblib` next to the Day 1 model, and `predict()` now uses it by default.

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

`predict()` accepts a single dict, a list of dicts, or a DataFrame, and returns one result per customer. It uses the Day 2 champion if it exists, otherwise the Day 1 model. Risk tiers are a presentation convenience: `high` ≥ 0.70, `medium` ≥ 0.40, otherwise `low`.

---

## Design notes

- **Simple code.** Short functions, plain loops and comments that explain *why*, so a beginner can read every file top to bottom.
- **Baseline first.** Logistic regression is deliberately simple: fast, interpretable, and a fair reference point for the stronger models coming next.
- **No leakage.** Cleaning and splitting happen before fitting; all encoding/scaling is learned inside the pipeline on training data only.
- **Reproducible.** Fixed `random_state=42` and a stratified split preserve the churn rate in both folds.
- **Persist the whole pipeline**, not just the estimator — preprocessing travels with the model.
- **Class imbalance is visible.** At 26.5% churn, accuracy alone flatters the model, so precision, recall and ROC-AUC are reported alongside it.

---

## Roadmap

- [x] **Day 1** — End-to-end churn pipeline + persisted ML service + baseline metrics
- [x] **Day 2** — Compare Decision Tree, Random Forest and SVM; Random Forest is the champion
- [ ] Later — feature engineering, class-imbalance handling, API layer, monitoring, deployment

---

## Tech stack

`pandas` · `scikit-learn` · `joblib` — Python 3.11+

Built as part of the **RetailMind AI** 35-day incremental project.
