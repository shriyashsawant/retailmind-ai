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
│   ├── train_day6.py    # Day 6: CNN that classifies product images (Fashion-MNIST + real photos if present)
│   ├── catalog_service.py   # tag pictures with the saved image classifier
│   ├── train_day7.py    # Day 7: pretrained ResNet50 vs the Day 6 CNN
│   ├── detector.py      # Day 7: find objects and their boxes with YOLO (optional: needs ultralytics)
│   ├── segmentation.py  # Day 8: cut the product out of its picture (OpenCV)
│   ├── explain.py       # Day 8: Grad-CAM heatmaps
│   ├── vision.py        # Day 8: one object that packages classify + segment + explain (+ detect)
│   ├── run_day8.py      # Day 8: measure the above and save pictures
│   ├── demand_data.py   # Days 9-10: daily store sales (real Rossmann file or made-up) cut into windows
│   ├── train_day9.py    # Day 9: LSTM that forecasts tomorrow's sales, vs simple baselines
│   ├── check_day9_seeds.py    # Day 9: repeat the training from 5 random starts to see how much scores swing
│   ├── train_day10.py   # Day 10: Seq2Seq + attention that forecasts the next 4 weeks
│   ├── check_day10_seeds.py   # Day 10: repeat the training from 5 random starts to see how much scores swing
│   ├── train_day11.py   # Day 11: a VAE draws extra pictures of a thin category; does the CNN improve?
│   ├── demand_service.py   # the demand module: forecast_next_day() (Day 9), forecast_weeks() (Day 10)
│   └── service.py       # predict() helper + command line
├── docs/
│   └── LEARNING_GUIDE.md   # Days 1-11 explained in plain English
├── models/              # persisted pipeline + metrics (generated)
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_day5_keras_ann.ipynb   # Day 5: the ANN increment with saved outputs
│   ├── 03_day6_cnn_catalog.ipynb # Day 6: the CNN increment with saved outputs
│   ├── 04_day7_transfer_detection.ipynb   # Day 7: transfer learning + detection results
│   ├── 05_day8_segmentation_gradcam.ipynb # Day 8: background removal + Grad-CAM, run live
│   ├── 06_day9_lstm_demand.ipynb          # Day 9: LSTM demand forecast, results and the demand module
│   ├── 07_day10_seq2seq_attention.ipynb   # Day 10: 4-week forecast with Seq2Seq + attention
│   └── 08_day11_vae_augmentation.ipynb    # Day 11: VAE pictures for a thin category, results and samples
├── reports/             # generated data profile, Day 7 detections, Day 8 and Day 11 pictures, Day 10 attention
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

# Day 6: CNN that classifies product images (about 8 minutes on a laptop CPU)
python -m src.train_day6
python -m src.catalog_service

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
- `models/catalog_cnn.keras` + `models/day6_results.json` — the Day 6 image classifier and its results
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

## Day 6 — A CNN that classifies product images

**Question:** can we tag catalog pictures automatically, and does a CNN beat a plain dense network on images?

`python -m src.train_day6` does this:

1. **Prototype on Fashion-MNIST** (60,000 training and 10,000 test pictures, 28×28 grey, 10 categories; built into Keras). Three designs are trained on the same pictures: a **dense** network (flattened pixels), a **CNN** (conv → pool → conv → pool → dropout → dense) and the same CNN with **augmentation** (random flips, small turns and zooms, applied only while training).
2. **Early stopping** on validation loss (up to 40 rounds, stop after 5 without improvement). The design is chosen by validation loss, never by the test pictures.
3. **Confusion matrix** on the 10,000 test pictures for the chosen model.
4. **Saves the catalog classifier** as `models/catalog_cnn.keras`, the first computer-vision module. `python -m src.catalog_service` tags sample pictures with it.
5. **Real product photos (optional).** If the Kaggle *Fashion Product Images* are in `data/fashion_product_images/` (`styles.csv` + `images/`), the same CNN design is trained on them (pictures loaded from disk, resized to 64×64, 10 most common article types, augmentation on).

| Model | Weights | Rounds (best) | Validation loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dense network (flattened pixels) | 101,770 | 21 (16) | 0.3216 | 0.8824 |
| **CNN, no augmentation** — chosen | 121,930 | 28 (23) | **0.2175** | **0.9163** |
| CNN with augmentation | 121,930 | 40 (36) | 0.2541 | 0.9027 |

**The CNN beats the dense network by 3.4 points of accuracy** with a similar number of weights, because convolution filters look at neighbouring pixels (edges, then shapes) instead of treating the picture as a flat list. **Augmentation did not help on Fashion-MNIST** (0.9027 vs 0.9163). These pictures are already centred and the same size, so random flips and turns mostly add noise, and the augmented network used all 40 allowed rounds, so it was still creeping up. Augmentation is kept for the real-photo pipeline, where pictures are messier.

**Which categories are confused?** Recall per category for the chosen CNN: Trouser 0.979, Bag 0.981, Sandal 0.988 and Ankle boot 0.963 are almost always right, while **Shirt is weakest at 0.731**, then Coat 0.855 and T-shirt/top 0.878. The biggest mix-ups are Shirt called T-shirt/top (123 times), T-shirt/top called Shirt (81), Coat called Pullover (66) and Shirt called Pullover or Coat (58 each). These are tops that look alike at 28×28.

**Honest limits.**
- The real *Fashion Product Images* (Kaggle, needs an account to download) were **not** available, so **no real-photo accuracy is reported**. The loader and the real-photo CNN code were tested only on synthetic pictures to prove the code runs on the real folder layout (a broken `styles.csv` row and a missing photo are skipped).
- Each model was trained once with a fixed seed, so gaps of about 0.01 (for example CNN vs augmented CNN) are within what a different seed could change.
- Like Day 5, this needs TensorFlow (Python 3.9–3.12) and `pillow`.

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

## Day 7 — Transfer learning and object detection

```bash
# Transfer learning vs the Day 6 CNN (about 6 minutes on a laptop CPU; downloads ResNet50 weights, ~100 MB, once)
python -m src.train_day7

# Detection with a pretrained YOLO model (needs: pip install ultralytics, which also installs PyTorch)
python -m src.detector path/to/picture.jpg --save reports/boxes.jpg
```

**Transfer learning.** ResNet50 (learned on ImageNet) replaces the from-scratch CNN. First the base is frozen and only a new head is trained; then the top 20 layers are unfrozen and trained gently. Every model gets the **same 10,000 training pictures** and is scored on the **same 10,000 test pictures**.

| Model | Trainable weights | Test accuracy | Seconds |
| --- | ---: | ---: | ---: |
| Day 6 CNN, from scratch | 121,930 | **0.8774** | 18 |
| ResNet50, frozen (feature extraction) | 20,490 | 0.8266 | 144 |
| ResNet50, fine-tuned (top 20 layers) | 8,939,530 | 0.8656 | 175 |

For reference, the Day 6 CNN trained on all 60,000 pictures scores 0.9163.

**Honest reading: transfer learning did not win here.** The small from-scratch CNN beat ResNet50 on this data. Likely reasons (not tested): Fashion-MNIST pictures are tiny 28x28 greyscale, so enlarging them to 64x64 and copying the grey channel three times gives ResNet50 very little of the detail ImageNet features expect, while the Day 6 CNN was built for exactly this picture size. Fine-tuning helped (+0.039 over frozen) but did not close the gap. Single run, single seed; gaps of about 0.01 are noise. Transfer learning would be expected to pay off on real, full-size catalog photos (the Kaggle Fashion Product Images set), which were **not run** because that dataset needs a Kaggle account.

**Detection.** `src/detector.py` runs pretrained YOLOv8n (COCO, 80 everyday object types) and returns each object's label, confidence and box. Verified on the public Ultralytics sample picture (4 people and 1 bus found; see `reports/day7_detections.json` and `.jpg`). That sample is **not** a catalog image, and YOLO's COCO classes contain no fashion-specific categories, so on catalog photos it would find generic things like "person" or "handbag", not dress or shirt. Where detection helps: catalog QA (is the product present and centred?), pictures with several products, and shelf analytics. To locate specific fashion items it would need to be fine-tuned on a labelled fashion detection set, which is the next step. `ultralytics` is kept out of `requirements.txt` because it pulls in PyTorch; install it only if you use the detector.

---

## Day 8 — Background removal, Grad-CAM and the packaged CV module

```bash
pip install -r requirements.txt            # now includes OpenCV
python -m src.run_day8                     # measure + save reports/day8_catalog_grid.png and models/day8_results.json
python -m src.run_day8 --photo shoe.jpg    # also cut a product out of a colour photo (uses the saved detector box)
```

```python
from src.vision import CatalogVision
vision = CatalogVision()
result = vision.analyze(picture)           # grey 28x28, values 0-255
result["category"], result["confidence"], result["product_share"], result["heat_on_product"]
result["clean"]                            # the product on a white background
result["explanation"]                      # the Grad-CAM overlay for the catalog team
```

`src/vision.py` packages classification (Day 6 CNN), detection (Day 7 YOLO, optional) and the new segmentation (`src/segmentation.py`, OpenCV) and explanation (`src/explain.py`, Grad-CAM). `reports/day8_catalog_grid.png` shows, for 8 test pictures: original, mask, product on white, Grad-CAM.

**Segmentation method.** OpenCV, no model to download. Fashion-MNIST products sit on a flat background, so a pixel counts as product when it differs from the background colour (read from the picture edge) by more than 8 grey levels; enclosed holes are filled, background touching the edge (the gap between trouser legs) is kept. For colour photos, GrabCut runs inside a box from the detector. A learned segmenter (U-Net, Mask R-CNN) was **not** built, because it needs labelled masks we do not have.

| Check (1,000 test pictures) | Result |
| --- | ---: |
| Classifier accuracy, original pictures | 0.914 |
| Accuracy after background removal (put back on black) | 0.911 |
| Same answer before and after cleaning | 99.6% |
| Accuracy on the cleaned picture **on a white background** | 0.340 |
| Product's share of the picture (mean) | 0.483 |
| Grad-CAM heat that falls on the product (mean) | 0.468 |

**Honest reading**
- Segmentation here is an easy case. These pictures already have a flat black background, so "background removal" mostly confirms what is there. It is **not** evidence that it would work on real, messy catalog photos, which were not available (the Kaggle dataset needs an account). On the one colour photo tried (a public Ultralytics sample, not a product), GrabCut kept the bus and the people inside the box and removed the sky, but it cannot separate objects that overlap.
- My first version used Otsu thresholding and lost dark products such as coats (accuracy fell to 0.880), and a blur-and-close step filled the gap between trouser legs. Both are fixed; the threshold of 8 was chosen on **training** pictures (anything from 3 to 15 kept accuracy), not the test set.
- **Classify first, composite last.** The CNN was trained on black backgrounds, so on a white background it falls to 0.340. Use the white-background version only for publishing, never as classifier input.
- **Grad-CAM does not show the model "looking at the product".** The share of heat on the product (0.468) is about the same as the product's share of the picture (0.483), and it is similar for right (0.466) and wrong (0.483) answers. At this size the last convolution map is only 11x11, so the heat is coarse; for trousers it lights up the space beside the legs, i.e. the model uses shape and outline. So the heatmap shows *where the evidence is*, and a catalog reviewer should not read it as proof that a decision is right.
- One run, the first 1,000 test pictures. Differences of a few pictures (0.911 vs 0.914) are noise.

---

## Day 9 — An LSTM that forecasts demand

```bash
python -m src.train_day9        # about 1 minute on a laptop CPU
python -m src.demand_service    # tomorrow's forecast for every store
```

```python
from src.demand_service import forecast_next_day
forecast_next_day(last_28_days_of_sales)      # one store's last 28 days of sales -> tomorrow's sales
```

**The task.** Read a store's last 28 days of sales and predict the next day. The sales are cut into windows (28 days in, 1 day out) and split **by time**: the oldest days train the model, the 8 weeks before the end validate it, and the **latest 8 weeks** test it. Each window is scaled by its own average and spread, so one model serves small and big stores.

**Important: the data is made up.** The real Kaggle *Rossmann Store Sales* file needs a Kaggle account, so it was not used. `src/demand_data.py` loads `data/raw/rossmann/train.csv` if you put it there; otherwise it generates **made-up** daily sales for 8 stores (store size, a weekly pattern with Sundays closed, a yearly wave, slow growth, a promotion on 5 days of every 2 weeks, and noise). These scores show that the code works. They are **not** evidence about real demand.

| Forecast of the next day (448 test windows) | MAE (sales units) | RMSE |
| --- | ---: | ---: |
| Same as yesterday | 1991.1 | 3072.3 |
| Same as the same weekday last week | 665.2 | 853.1 |
| Same as the same weekday 2 weeks ago | 278.2 | 386.1 |
| **LSTM** (4,385 weights), one run (seed 42) | 248.3 | 329.9 |
| **LSTM**, mean of 5 random starts | **271.3** (std 30.5) | |

**Honest reading**
- **One run said the LSTM beats the best baseline by 11% (248.3 vs 278.2). Five random starts say it roughly ties it.** A neural network starts from random numbers, so repeating the training from 5 different starts (`python -m src.check_day9_seeds`) gave 238.3, 248.9, 256.4, 292.9 and 319.9: mean 271.3, spread (std) 30.5. That is only about 2.5% better than the baseline's 278.2, and the LSTM beat the baseline in just 3 of the 5 runs. The 248.3 in the table was a favourable start.
- The fair baseline is "2 weeks ago", because the made-up promotions repeat every 2 weeks; comparing only with "yesterday" or "last week" would have made the LSTM look far better than it is.
- The first training run stopped at the 40-round limit, so the limit was too low (the same lesson as Day 6). With a limit of 150, early stopping ended it at 69 rounds.
- The data was made up with a clean repeating pattern, so a model that finds it should do well. Real Rossmann sales (holidays, school breaks, competitors opening) are messier and may give different results.
- MAE is used because many days have 0 sales (closed), which breaks percentage errors. The baselines do not learn, so they give the same score every time.

---

## Day 10 — Seq2Seq with attention: forecasting 4 weeks ahead

```bash
python -m src.train_day10       # about 3 minutes on a laptop CPU (run src.train_day9 first)
python -m src.demand_service    # now also prints the next 4 weeks for every store
```

```python
from src.demand_service import forecast_weeks
forecast_weeks(last_56_days_of_sales)      # {"daily": [28 numbers], "weekly_totals": [4 numbers]}
```

**The task.** Read 8 weeks (56 days) of a store's sales and forecast the next 4 weeks (28 days) at once, with the same time-based split as Day 9 (232 test windows). A **Seq2Seq** model has an *encoder* that reads the history and a *decoder* that writes the forecast. **Attention** lets every forecast day look back across all 56 past days and choose which ones to listen to. The decoder is not fed its own earlier guesses (no teacher forcing, no step-by-step loop). The sales are **made up**, as on Day 9.

| Forecast of the next 28 days (one run, seed 42) | MAE | RMSE | MAE week 1 / 2 / 3 / 4 |
| --- | ---: | ---: | --- |
| Same weekday last week | 474.2 | 671.6 | 661.3 / 278.4 / 677.5 / 279.4 |
| Same weekday 2 weeks ago | 284.6 | 403.3 | 278.0 / 278.4 / 302.7 / 279.4 |
| Day 9 LSTM, rolled forward 28 times | 287.0 | 384.9 | 253.5 / 277.2 / 311.9 / 305.4 |
| Seq2Seq **without** attention | **272.2** | **367.7** | 251.8 / 266.1 / 279.1 / 291.7 |
| Seq2Seq **with** attention | 286.0 | 386.6 | 264.1 / 279.3 / 302.2 / 298.4 |

**Honest reading**
- **The table above is ONE random start (seed 42), and one start is not reliable.** A neural network starts from random numbers. Run 5 times (`python -m src.check_day10_seeds`), the score swings a lot:

  | 28-day MAE over 5 random starts | Mean | Spread (std) | Best to worst |
  | --- | ---: | ---: | --- |
  | Seq2Seq without attention | 401.0 | 132.4 | 287.1 to 645.8 |
  | Seq2Seq with attention | 315.2 | 44.4 | 275.6 to 397.9 |

  Attention was better in 3 of the 5 runs, better by 85.8 on average, and far more stable. With only 5 runs this is suggestive, not proof. My first reading of this page said "attention did not help" because of the single seed-42 run (272.2 without vs 286.0 with); that was a lucky start for the no-attention model and the conclusion was wrong.
- **Neither Seq2Seq version reliably beats a trivial rule.** The best simple baseline, "same weekday 2 weeks ago", scores 284.6. The average Seq2Seq score (315.2 with attention, 401.0 without) is worse than that. Only some individual runs beat it. The saved model (seed 42, with attention) scores 286.0, about the same as the baseline.
- **Rolling Day 9 forward gets worse with distance** (253.5 in week 1, 305.4 in week 4), the "mistakes pile up" problem. It was run once, with the trained Day 9 model, so it has no seed spread of its own.
- **The attention weights do not give a clean explanation.** On average the forecast looks most at the days 29 to 33 days back. The share of attention on 7, 14, 21 ... days back is 0.152, barely above an even spread (0.143), so it did not learn "same weekday" lags. In `reports/day10_attention.png` the bright band moves diagonally: later forecast days look at more recent past days. Attention here is over the encoder's LSTM states, and each state mixes in everything before it, so a bright cell does not mean that single day's sales mattered (a likely reason, not tested).
- The data is made up, so these scores show that the code works, not how it would do on real Rossmann sales. If you add the real `train.csv`, rerun both scripts.

---

## Day 11 — A VAE that draws extra pictures for a thin category

```bash
python -m src.train_day11       # about 15 minutes on a laptop CPU (run src.train_day6 first)
```

**The task.** Some categories have few training pictures and the classifier is weakest on them. A **VAE** (variational autoencoder) is a network that learns to draw new pictures that look like its training pictures. The script finds the weakest category in the Day 6 results (**Shirt**, recall 0.731), then **pretends we only have 100 shirt pictures** (and 1,000 of every other category). It trains a VAE on those 100 shirts, draws 900 new ones, adds them to the training set, retrains the Day 6 CNN, and compares per-category recall on the full 10,000-picture test set. Every number below is the average of **5 random starts**, because Day 9 and Day 10 showed that one run can mislead.

| Training set | Shirt recall (mean ± std) | Accuracy | Other 9 categories (mean recall) | T-shirt recall |
| --- | ---: | ---: | ---: | ---: |
| Thin data only (100 shirts) | 0.082 ± 0.083 | 0.8411 | 0.9254 | 0.9058 |
| + the 100 real shirts **repeated** to 1,000 (control) | **0.273 ± 0.053** | **0.8602** | 0.9255 | 0.9106 |
| + 900 **VAE** pictures | 0.185 ± 0.037 | 0.8507 | 0.9247 | 0.9022 |

For reference, the Day 6 CNN trained on all 60,000 pictures (6,000 shirts) reached Shirt recall 0.731. `reports/day11_samples.png` shows 24 real shirts (top) and 48 VAE drawings (below).

**Honest reading**
- **The VAE pictures helped, but plain repetition helped more.** With only 100 shirts the classifier almost never says "Shirt" (recall 0.08). Adding the VAE's pictures lifts that to 0.19 without hurting the other categories (0.925 either way; T-shirt recall dipped by 0.004). But simply repeating the 100 real pictures to the same count gives 0.27, and it beat the VAE in all 5 pairs of runs that share the same random start. So in this experiment the VAE did **not** earn its place over a free baseline.
- **Why, probably (not tested):** the VAE's drawings are blurry and look alike. In `day11_samples.png` they are all an "average long-sleeve shirt", with none of the tank tops or bold patterns of the real ones, and the numbers agree (variety between drawings 6.8 against 9.5 between real shirts). They add little new information, while repeated real pictures keep every sharp detail.
- **My "keep only realistic outputs" step did not filter anything.** I compared each drawing with its nearest real shirt and dropped those farther than 95% of real pictures are from each other; all 900 passed. Blurry average pictures sit close to real ones in raw pixel distance, so this check cannot tell blurry from realistic. A better check (for example a classifier trained to tell real from drawn) was not built.
- **Nothing here comes close to having real data.** The best setup reaches Shirt recall 0.27, against 0.73 with 60 times more shirts.
- **Limits.** Shirt is not really rare in Fashion-MNIST; it was thinned on purpose to simulate a rare category. The Kaggle product photos were not used (they need an account). It is one small dense VAE (trained once, seed 42), so the spread between different VAE trainings is not measured, and a better generator (a convolutional VAE or a GAN) might do better. Not tried.

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
- [x] **Day 6** — CNN image classifier for catalog photos (Fashion-MNIST prototype; real product photos supported but not yet run)
- [x] **Day 7** — ResNet50 transfer learning compared with the Day 6 CNN (the CNN won on Fashion-MNIST), plus a YOLO detector module
- [x] **Day 8** — OpenCV background removal, Grad-CAM explanations and one packaged CV module (closes Phase 2)
- [x] **Day 9** — LSTM demand forecast (next day) on made-up Rossmann-style sales; over 5 random starts it roughly ties the best simple baseline
- [x] **Day 10** — Seq2Seq + attention 4-week forecast. Over 5 random starts attention was more stable and better on average, but neither version reliably beat a "same weekday 2 weeks ago" baseline; the demand module now emits multi-week forecasts
- [x] **Day 11** — VAE pictures for a thin category (Shirt): they lifted Shirt recall from 0.08 to 0.19, but repeating the 100 real pictures did better (0.27)
- [ ] Later — class-imbalance handling, API layer, monitoring, deployment

---

## Tech stack

`pandas` · `scikit-learn` · `joblib` · `xgboost` · `tensorflow` / `keras` (Day 5) — Python 3.11 or 3.12

Built as part of the **RetailMind AI** 35-day incremental project.
