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
│   ├── train_all.py     # the full fair comparison (12 models x 3 setups, all tuned)
│   ├── regression_demo.py  # predict a number (monthly bill) with Linear Regression
│   ├── train_day4.py    # Day 4: features, PCA, bagging + boosting, final model
│   ├── train_day5.py    # Day 5: Keras neural network vs the best classical model
│   ├── ann_service.py   # score customers with the saved neural network
│   └── service.py       # predict() helper + command line
├── docs/
│   └── LEARNING_GUIDE.md   # Days 1-5 explained in plain English
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

# Full fair comparison: every model x every setup, each one tuned (about 15-30 minutes)
python -m src.train_all

# Day 5: Keras neural network vs the best classical model (needs TensorFlow, Python 3.9-3.12)
python -m src.train_day5
python -m src.ann_service

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
- `models/churn_best_overall.joblib` + `models/all_models_results.json` — the winner of the full fair comparison
- `models/churn_ann.keras` + `models/churn_ann_prepare.joblib` + `models/day5_results.json` — the Day 5 neural network, its data-preparation step and its results
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
| **Random Forest (Day 4 step-by-step pick)** | bagging | **0.6349** | **0.7968** | 0.5265 | 0.6340 | 0.8442 |
| Extra Trees | bagging | 0.6293 | 0.7914 | 0.5193 | 0.6271 | 0.8404 |
| Bagging (plain trees) | bagging | 0.6346 | 0.7727 | 0.5402 | 0.6359 | 0.8433 |
| AdaBoost | boosting | 0.6169 | 0.7460 | 0.5439 | 0.6291 | 0.8150 |
| Gradient Boosting | boosting | 0.5853 | 0.5321 | 0.6746 | 0.5949 | 0.8459 |
| Hist Gradient Boosting | boosting | 0.6285 | 0.7807 | 0.5290 | 0.6307 | 0.8423 |
| XGBoost | boosting | 0.6303 | 0.7834 | 0.5149 | 0.6214 | 0.8452 |

In this step Random Forest had the best validation F1 (chosen without looking at the test customers). This step used untuned settings, so see the full fair comparison below for the final answer. The truth is that **Random Forest, Bagging, Hist Gradient Boosting and XGBoost are practically tied**: their F1 scores are within about 0.01 of each other, which is smaller than the normal noise of a 1,409-customer test set. The models without a `class_weight` option (Gradient Boosting) flag fewer churners at the 0.50 cut-off (recall 0.53) even though their ROC-AUC is among the best; moving the cut-off is a later improvement.

XGBoost needs `pip install xgboost`, and on a Mac also `brew install libomp`. If it is missing, `train_day4.py` skips it and carries on.

### The full, fair comparison (`python -m src.train_all`)

The steps above tested different models in different ways. So `src/train_all.py` runs **every one of 12 models in all 3 setups** (original columns, new features, new features + PCA), and **each one is tuned** with `GridSearchCV` (5-fold cross-validation, F1). It also scores each model with default settings so we can see whether tuning helped. The winner is chosen by validation F1 on the training customers, never by the test customers.

Tuned validation F1 (higher is better; the best setup for each model is marked with *):

| Model | Original columns | New features | New features + PCA |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.6326 * | 0.6298 | 0.6263 |
| K-Nearest Neighbours | 0.6056 * | 0.6039 | 0.5997 |
| Naive Bayes | 0.5956 | 0.5965 | 0.6251 * |
| SVM | 0.6233 | 0.6252 * | 0.6214 |
| Decision Tree | 0.6211 * | 0.6185 | 0.6151 |
| Random Forest | 0.6370 * | 0.6369 | 0.6288 |
| Extra Trees | 0.6256 | 0.6297 * | 0.6247 |
| **Bagging** | **0.6387 \*** | 0.6372 | 0.6328 |
| AdaBoost | 0.6217 * | 0.6169 | 0.6163 |
| Gradient Boosting | 0.5906 * | 0.5820 | 0.5684 |
| Hist Gradient Boosting | 0.6356 * | 0.6339 | 0.6279 |
| XGBoost | 0.6322 * | 0.6298 | 0.6254 |

What we learned:

| Question | Answer |
| --- | --- |
| Did tuning help? | **Yes**, in 31 of 36 cases (average +0.039 F1). It helped most where the default settings overfit: Decision Tree +0.129, Extra Trees +0.107. It barely mattered for Logistic Regression, SVM and Gradient Boosting (under +0.003). |
| Did the new features help? | **No, not really.** They helped 3 of 12 models and on average slightly hurt (−0.0016). The honest conclusion is that these 6 features add nothing the models had not already found in the original columns. |
| Did PCA help? | **Almost never.** It helped only Naive Bayes (+0.029, because PCA removes the overlap that Naive Bayes dislikes) and slightly hurt the other 11. |

**Overall winner: Bagging on the original columns** (validation F1 0.6387, test: recall 0.7727, precision 0.5463, F1 0.6401, ROC-AUC 0.8462), saved as `models/churn_best_overall.joblib` and used by the service. Be careful how you read this: Bagging, Random Forest (0.6370) and Hist Gradient Boosting (0.6356) are separated by about 0.003, which is smaller than the normal noise of this test, so they are effectively tied. XGBoost needs `pip install xgboost` (on a Mac also `brew install libomp`); if it is missing the script skips it.

Test customers, each model in its own best setup:

| Model | Best setup | Recall | Precision | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression | original | 0.7861 | 0.5087 | 0.6176 | 0.8410 |
| K-Nearest Neighbours | original | 0.6016 | 0.6131 | 0.6073 | 0.8285 |
| Naive Bayes | new + PCA | 0.7166 | 0.5164 | 0.6002 | 0.8144 |
| SVM | new features | 0.6444 | 0.5980 | 0.6203 | 0.8270 |
| Decision Tree | original | 0.7594 | 0.5259 | 0.6214 | 0.8350 |
| Random Forest | original | 0.7968 | 0.5165 | 0.6267 | 0.8440 |
| Extra Trees | new features | 0.7460 | 0.5225 | 0.6145 | 0.8370 |
| **Bagging** | original | 0.7727 | 0.5463 | **0.6401** | **0.8462** |
| AdaBoost | original | 0.7807 | 0.5114 | 0.6180 | 0.8292 |
| Gradient Boosting | original | 0.5000 | 0.6655 | 0.5710 | 0.8453 |
| Hist Gradient Boosting | original | 0.7807 | 0.5242 | 0.6273 | 0.8438 |
| XGBoost | original | 0.7807 | 0.5233 | 0.6266 | 0.8376 |

### Regression demo (predicting a number)

Churn is a **classification** problem (yes / no). **Regression** predicts a number. `python -m src.regression_demo` predicts each customer's monthly bill with Linear Regression, Ridge Regression and a Random Forest Regressor (average error about 0.8 on an average bill of 64.76, R² 0.999). It is easy because a customer's bill is almost the sum of the services they chose.

Every saved model file contains the data preparation (scaling and one-hot) **and** the model, so serving prepares data exactly like training. (`models/churn_final.joblib` is the Day 4 step-by-step result; `models/churn_best_overall.joblib` is the winner of the full fair comparison above and is what the service uses.)

### Phase 1 summary — the baseline table

Same 1,409 test customers for every row, threshold 0.50.

| Day | Model | Recall | Precision | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Logistic Regression | 0.5588 | 0.6572 | 0.6040 | 0.8421 |
| 2 | Random Forest | 0.7914 | 0.5258 | 0.6318 | 0.8407 |
| 3 | Random Forest, tuned | 0.7968 | 0.5165 | 0.6267 | 0.8440 |
| 4 | Bagging, tuned (winner of the full comparison) | 0.7727 | 0.5463 | 0.6401 | 0.8462 |

The big jump was Day 2 (recall 0.56 → 0.79). Days 3 and 4 gave only small gains, and the Day 4 features and PCA did not help on this data. Every later day must beat **F1 0.6401** and **ROC-AUC 0.8462**.

---

## Day 5 — A Keras neural network (ANN) vs the classical champion

**Question:** does a neural network (depth) beat the best classical model on this table of customers?

`python -m src.train_day5` does five things:

1. **Warm-up on MNIST** (handwritten digits): a tiny network with one hidden layer reaches about 90% accuracy on unseen digits after 3 rounds. This is only to see layers, activations and a training loop working.
2. **Builds two churn networks** on the same customers and the same 45 prepared columns (scaled numbers + one-hot text): a **shallow** one (1 hidden layer of 16 neurons, 753 weights) and a **deeper** one (3 hidden layers of 64, 32, 16 neurons with 30% dropout, 5,569 weights). Both end in one sigmoid neuron, the churn probability.
3. **Trains with early stopping.** It watches the loss on a validation slice (20% of the training customers) and stops after 10 rounds with no improvement, then goes back to the best round (shallow: round 15 of 25, deeper: round 7 of 17). Because only about 1 in 4 customers churns, churners are weighted 2.77 times, the same idea as `class_weight="balanced"` on Day 2.
4. **Picks the network by validation loss**, never by the test customers (the deeper one: 0.5101 vs 0.5251).
5. **Benchmarks it** against the Phase 1 champion (Bagging) on the same 1,409 test customers, threshold 0.50, then saves the network (`models/churn_ann.keras`) and its preparation step.

| Model | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| **Best classical model (Bagging, Phase 1)** | 0.7727 | 0.5463 | **0.6401** | **0.8462** |
| Shallow ANN (1 hidden layer) | 0.7834 | 0.5060 | 0.6149 | 0.8402 |
| Deeper ANN (3 hidden layers) — chosen by validation loss | 0.7460 | 0.5186 | 0.6118 | 0.8389 |

**Did depth help? No, not on this data.** The chosen ANN's F1 is 0.028 lower than the classical model's and its ROC-AUC is 0.007 lower. The deeper network was also not better than the shallow one on the test customers (F1 0.6118 vs 0.6149), although it had a slightly lower validation loss. Treat the ANN and the classical models as roughly comparable (the ANN is a little behind), and note that the ANN scores come from one training run with one random seed, so the small gaps between the two networks are not solid evidence.

### When to use which

| Situation | Better choice | Why |
| --- | --- | --- |
| A table of customers like this one (about 7,000 rows, a mix of numbers and categories) | A classical tree-based or linear model | It matched or beat the ANN here, trains in seconds and needs no special hardware |
| You must explain a prediction to a business user | A classical model | Logistic Regression weights and tree rules can be read. A network's weights cannot |
| Images, audio, free text, or very large data | A neural network | It learns its own features. A table of clean columns has nothing for it to discover (Day 6 moves to product photos with a CNN) |
| Small data | A classical model | Networks need many examples, or they memorise (that is why early stopping and dropout were needed here) |
| Serving cost and simplicity matter | A classical model | One small `joblib` file. The ANN needs TensorFlow installed to serve |

So for churn on this table the classical champion stays the production model, and the ANN is kept as a benchmark that proves the point. Both can be served: `python -m src.service` (classical) and `python -m src.ann_service` (neural network).

**Environment note.** TensorFlow supports Python 3.9 to 3.12 (check the TensorFlow install page for newer versions). The code uses `import keras` (Keras 3), so it does not depend on the TensorFlow-specific API.

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
- [x] **Day 5** — Keras ANN benchmarked against the classical champion (the classical model stays in production)
- [ ] Later — CNN for product photos, class-imbalance handling, API layer, monitoring, deployment

---

## Tech stack

`pandas` · `scikit-learn` · `joblib` · `xgboost` · `tensorflow` / `keras` (Day 5) — Python 3.11 or 3.12

Built as part of the **RetailMind AI** 35-day incremental project.
