"""Day 3: tune six different models, check for underfitting / overfitting, and lock the best one.

Run from the main project folder:   python -m src.train_day3
"""
import json
import os
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import DAY3_MODEL_FILE, DAY3_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE
from src.data import get_train_test_data
from src.evaluate import evaluate
from src.pipeline import make_pipeline

SCORE = "f1"   # tuning goal: a good balance of catching churners (recall) and being right (precision)


def diagnose(train_score, validation_score):
    """Read bias (underfit) or variance (overfit) from two scores."""
    gap = train_score - validation_score
    if gap > 0.10:
        return "overfitting (high variance): it learns the training customers too well"
    if validation_score < 0.55:
        return "underfitting (high bias): too simple to find the pattern"
    return "balanced: small gap and a reasonable score"


def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    # For each model: the model, and the settings we want to try for it.
    # "model__" means "this setting belongs to the model step of the pipeline".
    # class_weight="balanced" is used wherever a model has it (see Day 2). K-Nearest
    # Neighbours and Naive Bayes do not have it, so they will tend to catch fewer churners.
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "K-Nearest Neighbours": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 15, 31, 51], "model__weights": ["uniform", "distance"]}),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.3, 1, 3]}),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [1, 20, 50]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 300], "model__max_depth": [5, 8, 12],
             "model__min_samples_leaf": [1, 10]}),
    }

    # 1. Try every combination of settings with 5-fold cross-validation.
    #    (The training customers are split into 5 parts; each part takes a turn as the "validation" part.)
    results = {}
    searches = {}
    for name, (model, settings) in candidates.items():
        print("Tuning", name, "...")
        pipeline = make_pipeline(X_train, model)
        search = GridSearchCV(pipeline, settings, cv=5, scoring=SCORE, return_train_score=True, n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search

        best = search.best_index_
        train_score = search.cv_results_["mean_train_score"][best]
        validation_score = search.cv_results_["mean_test_score"][best]
        results[name] = {
            "best_settings": {key.replace("model__", ""): value for key, value in search.best_params_.items()},
            "train_score": round(float(train_score), 4),
            "validation_score": round(float(validation_score), 4),
            "diagnosis": diagnose(train_score, validation_score),
        }

    # 2. The champion is the model with the best validation score (chosen WITHOUT looking at the test set).
    champion = None
    for name in results:
        if champion is None or results[name]["validation_score"] > results[champion]["validation_score"]:
            champion = name

    # 3. Learning curve: how do the scores change as the model sees more customers?
    sizes, train_scores, validation_scores = learning_curve(
        searches[champion].best_estimator_, X_train, y_train,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv=5, scoring=SCORE, n_jobs=-1)
    curve = []
    for i in range(len(sizes)):
        curve.append({"customers": int(sizes[i]),
                      "train_score": round(float(train_scores[i].mean()), 4),
                      "validation_score": round(float(validation_scores[i].mean()), 4)})

    last, before_last = curve[-1], curve[-2]
    if last["train_score"] - last["validation_score"] > 0.10:
        advice = "The gap is large, so use more regularisation (simpler settings) or get more data."
    elif last["validation_score"] < 0.55:
        advice = "Both scores are low, so add better features (Day 4)."
    elif last["validation_score"] - before_last["validation_score"] > 0.005:
        advice = "The validation score is still rising, so more data would help."
    else:
        advice = "The validation score has levelled off, so more data alone will not help; better features might."

    # 4. Only now look at the test customers (once, for every tuned model).
    test_scores = {}
    for name in results:
        test_scores[name] = evaluate(searches[name].best_estimator_, X_test, y_test)

    # ---- show everything ----
    print()
    print(f"{'Model':<22}{'Train F1':>10}{'Valid F1':>10}   Diagnosis")
    print("-" * 95)
    for name, r in results.items():
        mark = "  <- champion" if name == champion else ""
        print(f"{name:<22}{r['train_score']:>10}{r['validation_score']:>10}   {r['diagnosis']}{mark}")
    print()
    for name, r in results.items():
        print(name, "best settings:", r["best_settings"])
    print()
    print("Learning curve for", champion)
    for row in curve:
        print(f"  {row['customers']:>5} customers   train {row['train_score']}   validation {row['validation_score']}")
    print("Advice:", advice)
    print()
    print(f"{'Model':<22}{'Recall':>8}{'Precision':>11}{'F1':>8}{'ROC-AUC':>9}   (test customers)")
    for name, s in test_scores.items():
        print(f"{name:<22}{s['recall']:>8}{s['precision']:>11}{s['f1']:>8}{s['roc_auc']:>9}")

    # 5. Lock the champion: save the tuned pipeline and write down what we found.
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(searches[champion].best_estimator_, DAY3_MODEL_FILE)
    with open(DAY3_RESULTS_FILE, "w") as f:
        json.dump({"trained_at": datetime.now().isoformat(timespec="seconds"), "scoring": SCORE,
                   "champion": champion, "models": results, "learning_curve": curve,
                   "advice": advice, "test_scores": test_scores}, f, indent=2)
    print()
    print("Saved champion ->", DAY3_MODEL_FILE)
    print("Saved results  ->", DAY3_RESULTS_FILE)


if __name__ == "__main__":
    main()
