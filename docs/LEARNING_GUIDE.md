# RetailMind learning guide: Days 1 to 4, from the beginning

This guide explains what the code does and **why**, in plain English. Read it with the code open next to it. Every "file" name below is in the `src/` folder.

---

## 0. The problem: which customers are about to leave?

A phone company has 7,043 customers. About **1 in 4 (26.5%)** left ("churned"). If the company could spot who is about to leave, it could call them with an offer. So we build a **model**: a program that looks at a customer's details and guesses "will leave" or "will stay".

How does a model learn? Like a student with past exam papers:

1. Show it many past customers **with the answer** (left / stayed). This is **training**.
2. Hide the answers for other customers and ask it to guess. This is **testing**.
3. Count how often it is right.

---

## 1. Day 1: the baseline

**Goal:** the simplest model that works end to end, so every later day has something to beat.

### Step 1: Load and clean the data (`data.py`)
- The data is a table: one row per customer, columns like `tenure` (months), `Contract`, `MonthlyCharges`, and the answer column `Churn` ("Yes"/"No").
- Models only understand numbers, so `Churn` becomes `1` for Yes and `0` for No.
- `TotalCharges` was stored as text and 11 brand-new customers had it blank. We turn it into a number and make blanks `0` (they have not paid anything yet).

### Step 2: Split into training and test (`get_train_test_data`)
- 80% of customers (5,634) are for **training**. 20% (1,409) are **locked away** for the final test.
- Why lock some away? A student who memorises the answers to the practice paper looks brilliant on that paper but fails a new one. The test customers are the "new paper".
- `stratify=y` keeps the same share of leavers in both parts.
- `random_state=42` makes the split the same every time, so results can be compared.

### Step 3: Prepare the columns (`pipeline.py`)
- **Number columns** (tenure, MonthlyCharges...) get **scaled** (`StandardScaler`) so a column like TotalCharges (up to 8,000) does not drown out SeniorCitizen (0 or 1).
- **Text columns** (Contract, PaymentMethod...) get **one-hot encoded**: "Contract" becomes three 0/1 columns, "Is it month-to-month? Is it one-year? Is it two-year?". A model cannot do maths on the word "Month-to-month".
- Both steps and the model are joined in one `Pipeline`. This is important: the **same** preparation is applied when you train and when you predict later.

### Step 4: Train the model (`train.py`)
- We use **Logistic Regression**: it gives each column a weight, adds them up, and turns the total into a probability between 0 and 1 ("62% likely to leave"). If the probability is 0.5 or more we say "will leave".

### Step 5: Score it (`evaluate.py`). This is the part most people get wrong
Our results on the 1,409 test customers:

|  | Predicted stay | Predicted churn |
| --- | ---: | ---: |
| Actually stayed | 926 | 109 |
| Actually churned | 165 | 209 |

- **Accuracy 0.8055** = (926 + 209) / 1,409. Sounds good, but a lazy model that always says "stay" would already score about 0.73! Accuracy hides the problem.
- **Recall 0.5588** = 209 / (209 + 165). "Of the customers who really left, how many did we catch?" We caught only 56%. For this business, **recall is the number that matters**, because every missed leaver is a lost customer.
- **Precision 0.6572** = 209 / (209 + 109). "When we say someone will leave, how often are we right?"
- **F1** = one number that balances precision and recall (it is low if either is low).
- **ROC-AUC 0.8421** = "pick one leaver and one stayer at random: how often does the model give the leaver the higher probability?" 0.5 is a coin flip, 1.0 is perfect.

**Day 1 lesson:** a model can look good on accuracy and still miss almost half of the people you care about.

---

## 2. Day 2: try three stronger models

**Goal:** catch more leavers.

The three new models, in plain words:
- **Decision Tree:** a flowchart of yes/no questions. "Is the contract month-to-month? Yes → Is tenure under 12 months? Yes → likely to leave."
- **Random Forest:** hundreds of different trees that each vote. The crowd is wiser than one tree.
- **SVM:** draws the best dividing line (or curve) between the leavers and the stayers.

**The key trick: `class_weight="balanced"`.** Only 1 in 4 customers leaves, so a model can get a good score by mostly saying "stay". `balanced` tells it: "a leaver counts about three times as much as a stayer". It then tries harder to catch leavers.

**Fair comparison:** all models use the same customers, the same preparation (`make_pipeline`) and the same random seed. If anything differs, you cannot tell why one won.

**Result:** Random Forest won with recall **0.79** (Day 1 was 0.56). The price: precision fell from 0.66 to 0.53, so more loyal customers get flagged by mistake. That is a fair trade when a retention offer costs less than losing a customer.

---

## 3. Day 3: tuning, and "is it learning or memorising?"

### Hyperparameters (the model's settings)
A Random Forest has settings you choose before training, like how many trees (`n_estimators`) and how deep each tree may grow (`max_depth`). These are **hyperparameters**.

### GridSearchCV (try every combination)
`GridSearchCV` is "try every combination of settings, and tell me which is best". To judge fairly it uses **cross-validation**:

1. Split the 5,634 training customers into 5 parts.
2. Train on 4 parts, score on the 5th (the "validation" part). Repeat so every part gets a turn.
3. Average the 5 scores.

This gives a steadier score than one lucky split. The test customers are still untouched.

### Bias and variance: the most important idea of the day
Compare two scores: the **train score** (on customers the model studied) and the **validation score** (on customers it has not seen).

| Train score | Validation score | What it means | Called |
| --- | --- | --- | --- |
| high | much lower | memorising the practice paper | **overfitting** (high variance) |
| low | low | too simple to learn the pattern | **underfitting** (high bias) |
| high | about the same | learning properly | balanced |

We tuned six models: Logistic Regression, K-Nearest Neighbours, Naive Bayes, SVM, Decision Tree and Random Forest. All six came out balanced (no big gaps), and Random Forest had the best validation score. Its train F1 was 0.677 and its validation F1 0.637, a small gap.

Notice what tuning could **not** do: the six models ended up with validation F1 scores between 0.596 and 0.637. When many different models land in the same narrow band, the limit usually comes from the **data** (the clues we gave), not the model, which is the hint that leads to Day 4.

### Learning curve (should we collect more data?)
We train on 900, 1,800, ... 4,500 customers and watch the validation score. Ours stayed flat at about 0.63. So **more data would not help**. The model needs better information, which leads straight to Day 4.

---

## 4. Day 4: better features, PCA, and ensembles

### Features
A **feature** is one column of clues. We made 6 new ones in `features.py`, using our own business sense:
- tenure bucket (new customers leave more),
- charge ratio (did the price go up?),
- month-to-month flag (free to leave any time),
- has internet, number of services, pays automatically (customers who are more "tied in" stay).

**Honest result:** for Logistic Regression the new features did not help (F1 0.6322 to 0.6268), and for Random Forest they helped only a tiny bit (0.6341 to 0.6349). Not every good idea works, and measuring is how you find out.

### PCA (squash many columns into fewer)
After one-hot encoding we have 54 columns, many saying similar things. PCA combines them into 20 "mixed" columns that keep 95% of the information. The score barely changed (0.6254 vs 0.6268). But the new columns are blends that nobody can explain ("what does column 7 mean?"), so for this project we only noted it and did not use it.

### Bagging vs boosting
- **Bagging** (Random Forest): many trees trained **separately** on random pieces of the data, then they vote. This reduces overfitting.
- **Boosting** (Gradient Boosting, like XGBoost): trees are added **one after another**, and each new tree focuses on the mistakes of the previous ones.

**A fairness problem, and how we fixed it.** The steps above tested different models in different ways (some tuned, some not; some with the new features, some without). That is not a fair race. So `train_all.py` runs **every model (12 of them) in every setup (3 of them), each one tuned**, and also scores each one with its default settings. The three setups are: original columns, original + new features, and new features + PCA.

What that fair race showed:
- **Tuning helped a lot** (31 of 36 cases). Default Decision Trees and Extra Trees overfit badly until they are tuned (+0.13 and +0.11 F1). Logistic Regression barely changed, because it has few settings that matter.
- **The new features did NOT help**: only 3 of 12 models improved, and on average the score dropped a tiny bit (-0.0016). The models already found the same clues in the original columns. This is a real lesson: a clever idea is only good if the measurement says so.
- **PCA almost never helped** (1 of 12; it helped Naive Bayes because it removes the overlap Naive Bayes cannot handle).
- **Winner: Bagging on the original columns** (validation F1 0.6387). But Random Forest (0.6370) and Hist Gradient Boosting (0.6356) are within 0.003, so honestly the top three are a tie.

The seven ensembles in the first (uneven) test were: Random Forest, Extra Trees and Bagging (the bagging family), and AdaBoost, Gradient Boosting, Hist Gradient Boosting and XGBoost (the boosting family). Here is what that first test showed: Random Forest, Extra Trees and Bagging (the bagging family), and AdaBoost, Gradient Boosting, Hist Gradient Boosting and XGBoost (the boosting family).

**Result:** Random Forest was the final model (best validation F1 0.6349, test recall 0.797). But look at the table in the README: Random Forest, Bagging, Hist Gradient Boosting and XGBoost are **practically tied** (their F1 scores are within about 0.01). With only 1,409 test customers, a difference that small is noise, not proof. Gradient Boosting (the plain version has no `class_weight` setting) caught only 53% of leavers at the 0.5 cut-off even though its ranking (ROC-AUC) was among the best. A later day can fix that by moving the cut-off.

### Locking the model
`churn_best_overall.joblib` (the winner of the fair comparison) stores the preparation **and** the model together, so a new customer is prepared exactly like the training customers. The service also calls `add_features`, which builds the new columns for any saved model that needs them; the winner happens not to use them.

---

## 5. The model zoo: what each model is, in one or two lines

First, two families of problem:
- **Classification:** the answer is a group. "Will this customer leave: yes or no?" This is our project.
- **Regression:** the answer is a number. "What will this customer's monthly bill be?" See `src/regression_demo.py`.

| Model | Type | How it thinks | Good at | Watch out for |
| --- | --- | --- | --- | --- |
| **Linear Regression** | regression | draws the best straight line through the data and predicts a number from it | predicting numbers (a bill, a price) when the pattern is roughly a straight line | not for yes/no answers |
| **Logistic Regression** | classification | adds up weighted clues and turns the total into a probability | a fast, explainable baseline | cannot learn curved patterns on its own |
| **K-Nearest Neighbours** | classification | "find the 31 most similar customers; copy what most of them did" | simple, no training | slow on big data; needs scaled columns; no `class_weight` |
| **Naive Bayes** | classification | uses probability rules, assuming every clue is independent of the others | very fast, works with little data | the "independent clues" assumption is often wrong (here it flags many false alarms) |
| **SVM** | classification | finds the widest possible gap between the two groups | medium-size data with clear separation | slow to tune; hard to explain |
| **Decision Tree** | classification | a flowchart of yes/no questions | easy to read | one tree overfits easily |
| **Random Forest** | bagging | hundreds of different trees vote | strong all-rounder, hard to break | less readable than one tree |
| **Extra Trees** | bagging | like a Random Forest but splits are chosen more randomly | fast, sometimes less overfitting | slightly less accurate here |
| **Bagging** | bagging | many copies of one model, each trained on a random sample, then vote | calms down a jumpy model | needs a decent base model |
| **AdaBoost** | boosting | after each tree, give the customers it got wrong more weight | simple boosting | sensitive to noisy data |
| **Gradient Boosting** | boosting | each new tree predicts the **errors** of the trees before it | often very accurate | slower; easy to overfit if run too long |
| **Hist Gradient Boosting** | boosting | the same, but groups numbers into bins so it is much faster | bigger data | same care as Gradient Boosting |
| **XGBoost** | boosting | a fast, polished, popular version of Gradient Boosting | winning data competitions; handles missing values | extra install; many settings |

**Which one should you use?** There is no best model for every problem. You try several on the **same** data with the **same** fair test, and let the numbers decide, which is exactly what Days 2 to 4 did. On this data the tree-based models won by a small margin, and the simple Logistic Regression was surprisingly close.

---

## 6. The whole story in one table

| Day | Model | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: |
| 1 | Logistic Regression | 0.559 | 0.657 | 0.604 |
| 2 | Random Forest | 0.791 | 0.526 | 0.632 |
| 3 | Random Forest, tuned | 0.797 | 0.517 | 0.627 |
| 4 | Bagging, tuned (winner of the fair comparison) | 0.773 | 0.546 | 0.640 |

Notice: the **big** gain came from understanding the problem (use `balanced` so we catch leavers). The later days gave small steady gains. In real projects that is normal.

---

## 7. Words to remember

| Word | Meaning |
| --- | --- |
| Feature | one column of clues about a customer |
| Target | the answer column we predict (`Churn`) |
| Train / test split | study part vs locked-away exam part |
| Pipeline | preparation steps + model joined into one object |
| One-hot encoding | turning a text column into 0/1 columns |
| Scaling | putting number columns on a similar size |
| Recall | of the real leavers, how many we caught |
| Precision | of the people we flagged, how many really left |
| F1 | one number balancing precision and recall |
| ROC-AUC | how well the model ranks leavers above stayers |
| Hyperparameter | a setting chosen before training |
| Cross-validation | scoring on 5 different splits and averaging |
| Overfitting | great on training data, worse on new data |
| Underfitting | too simple to learn the pattern |
| PCA | squashing many columns into fewer |
| Bagging / boosting | trees that vote together / trees that fix each other's mistakes |

---

## 8. Try it yourself (best way to learn)

1. Run `python -m src.train` and open `models/metrics.json`. Find the four numbers of the confusion matrix and check recall by hand.
2. In `train_day2.py` remove `class_weight="balanced"` from the Random Forest and rerun. What happens to recall?
3. In `train_day3.py` add `20` to the `n_estimators` list and rerun. Did the best settings change?
4. In `features.py` add a new column of your own idea and look at `train_day4.py` Step 1 to see if it helps.
5. Change `THRESHOLD` in `config.py` from 0.5 to 0.4 and rerun `python -m src.train`. Recall goes up, precision goes down. Why? (Hint: a lower cut-off flags more customers as "will leave".)
