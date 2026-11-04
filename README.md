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
│   ├── train_day3.py    # Day 3: tune the models, check underfit / overfit
│   ├── features.py      # Day 4: the 6 new columns
│   ├── regression_demo.py  # predict a number (monthly bill) with Linear Regression
│   ├── train_day4.py    # Day 4: features, PCA, bagging + boosting, final model
│   └── service.py       # predict() helper + command line
├── docs/
│   └── LEARNING_GUIDE.md   # Days 1-4 explained in plain English
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

# Day 3: tune every model, check bias / variance, lock the champion
python -m src.train_day3

# Day 4: new features, PCA, bagging + boosting, lock the final model
python -m src.train_day4

# Bonus: predict a NUMBER (the monthly bill) with Linear Regression
python -m src.regression_demo

# Score the built-in demo customers (uses the newest model)
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
- `models/churn_v3.joblib` + `models/day3_tuning.json` — the Day 3 tuned champion and what we found
- `models/churn_final.joblib` + `models/day4_results.json` — the Day 4 final model and its results
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

## Day 3 — Tuning and bias / variance

Every model has settings (hyperparameters). We tried every combination with `GridSearchCV` and 5-fold cross-validation, scoring by **F1**. Then we compared the **train** score with the **validation** score: a big gap means overfitting (high variance), two low scores mean underfitting (high bias).

Six different kinds of model were tuned:

| Model | Best settings | Train F1 | Validation F1 | Diagnosis |
| --- | --- | ---: | ---: | --- |
| Logistic Regression | C = 0.1 | 0.6336 | 0.6326 | balanced |
| K-Nearest Neighbours | n_neighbors = 31, weights = uniform | 0.6254 | 0.6056 | balanced |
| Naive Bayes | var_smoothing = 0.001 | 0.5968 | 0.5956 | balanced |
| SVM | C = 0.3 | 0.6380 | 0.6233 | balanced |
| Decision Tree | max_depth = 5, min_samples_leaf = 50 | 0.6391 | 0.6211 | balanced |
| **Random Forest** | n_estimators = 100, max_depth = 8, min_samples_leaf = 10 | 0.6766 | **0.6370** | balanced (champion) |

Learning curve for the champion: the validation F1 stays around 0.63 from 900 up to 4,500 customers, so **more data alone will not help; better features might** (that is Day 4). The champion is saved as `models/churn_v3.joblib`.

On the 1,409 test customers (threshold 0.50):

| Model | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7861 | 0.5087 | 0.6176 | 0.8410 |
| K-Nearest Neighbours | 0.6016 | 0.6131 | 0.6073 | 0.8285 |
| Naive Bayes | **0.8369** | 0.4589 | 0.5928 | 0.8075 |
| SVM | 0.6364 | 0.5920 | 0.6134 | 0.8272 |
| Decision Tree | 0.7594 | 0.5259 | 0.6214 | 0.8350 |
| **Random Forest** | 0.7968 | 0.5165 | **0.6267** | **0.8440** |

Naive Bayes catches the most churners but is wrong more often, so its F1 and ROC-AUC are the lowest. Random Forest has the best balance, which is why it is the champion.

---

## Day 4 — Features, PCA and ensembles

We added 6 new columns (`src/features.py`): tenure bucket, charge ratio, month-to-month flag, has-internet flag, number of services, and pays-automatically flag.

| Check | Result |
| --- | --- |
| Do the new features help? (CV F1, same model with and without) | Logistic Regression 0.6322 → 0.6268 (no). Random Forest 0.6341 → 0.6349 (a tiny gain). |
| PCA (keep 95% of the information) | 54 columns → 20. CV F1 0.6254 vs 0.6268, so almost no loss, but the model becomes harder to explain, so we do not use it. |

Seven ensembles, all using the new features, compared with the Day 3 champion. The "Valid F1" column is the 5-fold cross-validation score on the training customers (this is what picks the final model); the other columns are on the test customers:

| Model | Family | Valid F1 | Recall | Precision | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Day 3 champion (old columns) | — | — | 0.7968 | 0.5165 | 0.6267 | 0.8440 |
| **Random Forest — final model** | bagging | **0.6349** | **0.7968** | 0.5265 | 0.6340 | 0.8442 |
| Extra Trees | bagging | 0.6293 | 0.7914 | 0.5193 | 0.6271 | 0.8404 |
| Bagging (plain trees) | bagging | 0.6346 | 0.7727 | 0.5402 | 0.6359 | 0.8433 |
| AdaBoost | boosting | 0.6169 | 0.7460 | 0.5439 | 0.6291 | 0.8150 |
| Gradient Boosting | boosting | 0.5853 | 0.5321 | 0.6746 | 0.5949 | 0.8459 |
| Hist Gradient Boosting | boosting | 0.6285 | 0.7807 | 0.5290 | 0.6307 | 0.8423 |
| XGBoost | boosting | 0.6303 | 0.7834 | 0.5149 | 0.6214 | 0.8452 |

Random Forest is the final model because it has the best validation F1 (it was chosen without looking at the test customers). The truth is that **Random Forest, Bagging, Hist Gradient Boosting and XGBoost are practically tied**: their F1 scores are within about 0.01 of each other, which is smaller than the normal noise of a 1,409-customer test set. The models without a `class_weight` option (Gradient Boosting) flag fewer churners at the 0.50 cut-off (recall 0.53) even though their ROC-AUC is among the best; moving the cut-off is a later improvement.

XGBoost needs `pip install xgboost`, and on a Mac also `brew install libomp`. If it is missing, `train_day4.py` skips it and carries on.

### Regression demo (predicting a number)

Churn is a **classification** problem (yes / no). **Regression** predicts a number. `python -m src.regression_demo` predicts each customer's monthly bill with Linear Regression, Ridge Regression and a Random Forest Regressor (average error about 0.8 on an average bill of 64.76, R² 0.999). It is easy because a customer's bill is almost the sum of the services they chose.

The final model is saved as `models/churn_final.joblib`. The saved file contains the data preparation (scaling and one-hot) **and** the model, and the new columns come from `src/features.py`, so serving prepares data exactly like training.

### Phase 1 summary — the baseline table

Same 1,409 test customers for every row, threshold 0.50.

| Day | Model | Recall | Precision | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Logistic Regression | 0.5588 | 0.6572 | 0.6040 | 0.8421 |
| 2 | Random Forest | 0.7914 | 0.5258 | 0.6318 | 0.8407 |
| 3 | Random Forest, tuned | 0.7968 | 0.5165 | 0.6267 | 0.8440 |
| 4 | Random Forest + new features | 0.7968 | 0.5265 | 0.6340 | 0.8442 |

The big jump was Day 2 (recall 0.56 → 0.79). Days 3 and 4 gave small steady gains. Every later day must beat **F1 0.6340** and **ROC-AUC 0.8442**.

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
- [x] **Day 3** — Tune every model with GridSearchCV, read bias / variance, lock the champion
- [x] **Day 4** — New features, PCA, bagging and boosting; final model and Phase 1 baseline table
- [ ] Later — deep learning (Keras ANN), class-imbalance handling, API layer, monitoring, deployment

---

## Tech stack

`pandas` · `scikit-learn` · `joblib` — Python 3.11+

Built as part of the **RetailMind AI** 35-day incremental project.
