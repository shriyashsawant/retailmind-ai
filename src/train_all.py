"""The full, fair comparison: EVERY model in EVERY setup, each one tuned.

The 3 setups are:
    1. original columns
    2. original columns + the 6 new features (Day 4)
    3. new features + PCA (fewer, blended columns)

For every model and setup we record:
    - the score with default settings
    - the score after tuning (GridSearchCV, 5-fold cross-validation on the training customers)
    - the score on the test customers (looked at once, only for reporting)

The overall winner is chosen by the tuned cross-validation score, never by the test score.

Run from the main project folder (takes roughly 15-30 minutes):   python -m src.train_all
"""
import json
import os
import warnings
from datetime import datetime

import joblib
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import ALL_RESULTS_FILE, BEST_MODEL_FILE, MODEL_FOLDER, RANDOM_STATE
from src.data import get_train_test_data
from src.evaluate import evaluate
from src.features import NEW_COLUMNS
from src.pipeline import make_pipeline

# XGBoost is a separate install (pip install xgboost; on a Mac also: brew install libomp).
# If it is missing we skip it and carry on with the other models.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings("ignore", category=FutureWarning)   # hides a harmless scikit-learn notice about SVC


def build_models(churn_weight):
    """Every model, and the settings we want to try for it. ("model__" = the model step of the pipeline.)"""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "K-Nearest Neighbours": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [15, 31, 51], "model__weights": ["uniform", "distance"]}),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.3, 1]}),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [20, 50]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 300], "model__max_depth": [5, 8, 12], "model__min_samples_leaf": [5, 10]}),
        "Extra Trees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__n_estimators": [200, 400], "model__max_depth": [8, 12], "model__min_samples_leaf": [5, 10]}),
        "Bagging": (
            BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6, class_weight="balanced"),
                              random_state=RANDOM_STATE),
            {"model__n_estimators": [50, 150], "model__max_samples": [0.5, 1.0]}),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
                               random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.1, 0.5, 1.0]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5], "model__max_iter": [100, 250]}),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(scale_pos_weight=churn_weight, eval_metric="logloss", n_jobs=1, random_state=RANDOM_STATE),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]})
    else:
        print("(XGBoost is not available here, so it is skipped.)")
    return models


def average(numbers):
    return sum(numbers) / len(numbers)


def main():
    X_train, X_test, y_train, y_test = get_train_test_data(new_features=True)
    old_columns = [c for c in X_train.columns if c not in NEW_COLUMNS]
    all_columns = list(X_train.columns)

    # name of the setup -> (which columns to use, whether to apply PCA)
    setups = {
        "original columns": (old_columns, False),
        "new features": (all_columns, False),
        "new features + PCA": (all_columns, True),
    }
    churn_weight = round(float((y_train == 0).sum() / (y_train == 1).sum()), 2)
    models = build_models(churn_weight)

    rows = []              # one row per (model, setup)
    best_f1 = -1.0
    best_pipeline = None
    best_label = None

    for model_name, (model, settings) in models.items():
        for setup_name, (columns, use_pca) in setups.items():
            print(f"[{len(rows) + 1}/{len(models) * len(setups)}] {model_name}  |  {setup_name}", flush=True)
            X_part = X_train[columns]

            # score with the default settings
            default_pipeline = make_pipeline(X_part, clone(model), use_pca)
            default_f1 = cross_val_score(default_pipeline, X_part, y_train, cv=5, scoring="f1", n_jobs=-1).mean()

            # tune: try every combination of settings
            pipeline = make_pipeline(X_part, clone(model), use_pca)
            search = GridSearchCV(pipeline, settings, cv=5, scoring="f1", n_jobs=-1)
            search.fit(X_part, y_train)
            tuned_f1 = search.best_score_

            # the test customers (for reporting only)
            test = evaluate(search.best_estimator_, X_test[columns], y_test)

            rows.append({
                "model": model_name,
                "setup": setup_name,
                "default_cv_f1": round(float(default_f1), 4),
                "tuned_cv_f1": round(float(tuned_f1), 4),
                "best_settings": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                "test_recall": test["recall"],
                "test_precision": test["precision"],
                "test_f1": test["f1"],
                "test_roc_auc": test["roc_auc"],
            })
            if tuned_f1 > best_f1:                      # keep the best one (chosen by validation, not test)
                best_f1 = tuned_f1
                best_pipeline = search.best_estimator_
                best_label = (model_name, setup_name)

    # ======================= show the results =======================
    def find(model_name, setup_name):
        for row in rows:
            if row["model"] == model_name and row["setup"] == setup_name:
                return row

    setup_names = list(setups)
    print("\n" + "=" * 100)
    print("A. Tuned validation F1 for every model in every setup (higher is better; * = best setup for that model)")
    print(f"{'Model':<24}" + "".join(f"{s:>22}" for s in setup_names))
    for model_name in models:
        scores = [find(model_name, s)["tuned_cv_f1"] for s in setup_names]
        cells = ""
        for s, score in zip(setup_names, scores):
            cells += f"{str(score) + (' *' if score == max(scores) else ''):>22}"
        print(f"{model_name:<24}{cells}")

    print("\nB. Did TUNING help? (tuned validation F1 minus default validation F1)")
    gains = [row["tuned_cv_f1"] - row["default_cv_f1"] for row in rows]
    better = len([g for g in gains if g > 0.0005])
    print(f"   average gain {average(gains):+.4f}; tuning helped in {better} of {len(rows)} cases")
    for model_name in models:
        model_gains = [find(model_name, s)["tuned_cv_f1"] - find(model_name, s)["default_cv_f1"] for s in setup_names]
        print(f"   {model_name:<24}average gain {average(model_gains):+.4f}")

    print("\nC. Did the NEW FEATURES help? (tuned F1 with new features minus original columns)")
    feature_gains = []
    for model_name in models:
        gain = find(model_name, "new features")["tuned_cv_f1"] - find(model_name, "original columns")["tuned_cv_f1"]
        feature_gains.append(gain)
        print(f"   {model_name:<24}{gain:+.4f}")
    print(f"   average {average(feature_gains):+.4f}; helped {len([g for g in feature_gains if g > 0.0005])} of {len(feature_gains)} models")

    print("\nD. Did PCA help? (tuned F1 with PCA minus the same new features without PCA)")
    pca_gains = []
    for model_name in models:
        gain = find(model_name, "new features + PCA")["tuned_cv_f1"] - find(model_name, "new features")["tuned_cv_f1"]
        pca_gains.append(gain)
        print(f"   {model_name:<24}{gain:+.4f}")
    print(f"   average {average(pca_gains):+.4f}; helped {len([g for g in pca_gains if g > 0.0005])} of {len(pca_gains)} models")

    print("\nE. Test customers: each model in its own best setup (the setup with the best validation F1)")
    print(f"{'Model':<24}{'Best setup':<22}{'Recall':>8}{'Precision':>11}{'F1':>8}{'ROC-AUC':>9}")
    for model_name in models:
        best_row = None
        for s in setup_names:
            row = find(model_name, s)
            if best_row is None or row["tuned_cv_f1"] > best_row["tuned_cv_f1"]:
                best_row = row
        mark = "  <- overall winner" if (model_name, best_row["setup"]) == best_label else ""
        print(f"{model_name:<24}{best_row['setup']:<22}{best_row['test_recall']:>8}{best_row['test_precision']:>11}"
              f"{best_row['test_f1']:>8}{best_row['test_roc_auc']:>9}{mark}")

    print(f"\nOverall winner (best validation F1 {round(float(best_f1), 4)}): {best_label[0]} with {best_label[1]}")

    # ======================= save =======================
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_FILE)
    with open(ALL_RESULTS_FILE, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "winner": {"model": best_label[0], "setup": best_label[1], "tuned_cv_f1": round(float(best_f1), 4)},
            "average_gain_from_tuning": round(average(gains), 4),
            "average_gain_from_new_features": round(average(feature_gains), 4),
            "average_gain_from_pca": round(average(pca_gains), 4),
            "results": rows,
        }, f, indent=2)
    print("Saved winner  ->", BEST_MODEL_FILE)
    print("Saved results ->", ALL_RESULTS_FILE)


if __name__ == "__main__":
    main()
