"""Day 4: new features, PCA, and many ensemble models (bagging + boosting). Then lock the final model.

Run Day 3 first (it makes the model we compare against), then from the main project folder:
    python -m src.train_day4
"""
import json
import os
from datetime import datetime

import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import DAY3_MODEL_FILE, DAY4_MODEL_FILE, DAY4_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE
from src.data import get_train_test_data
from src.evaluate import evaluate
from src.features import NEW_COLUMNS
from src.pipeline import make_pipeline, make_preprocessor

# XGBoost is a separate install (pip install xgboost). On a Mac it also needs the OpenMP
# library:  brew install libomp.  If it is missing we simply skip it and carry on.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def cv_scores(pipeline, X_train, y_train):
    """5-fold cross-validation on the training customers. Returns the average F1 and recall."""
    result = cross_validate(pipeline, X_train, y_train, cv=5, scoring=["f1", "recall"], n_jobs=-1)
    return {"f1": round(float(result["test_f1"].mean()), 4),
            "recall": round(float(result["test_recall"].mean()), 4)}


def main():
    # The same customers as before, but with the 6 new columns added.
    X_train, X_test, y_train, y_test = get_train_test_data(new_features=True)
    old_columns = [c for c in X_train.columns if c not in NEW_COLUMNS]

    def new_logistic():
        return LogisticRegression(max_iter=2000, class_weight="balanced")

    def new_forest():
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                      class_weight="balanced", random_state=RANDOM_STATE)

    # ---- Step 1: do the new features help? (same model, with and without them) ----
    print("Step 1: do the new features help?")
    feature_check = {}
    for name, make_model in [("Logistic Regression", new_logistic), ("Random Forest", new_forest)]:
        without = cv_scores(make_pipeline(X_train[old_columns], make_model()), X_train[old_columns], y_train)
        with_new = cv_scores(make_pipeline(X_train, make_model()), X_train, y_train)
        feature_check[name] = {"without_new_features": without, "with_new_features": with_new}
        print(f"  {name:<20} F1 without {without['f1']}  ->  with {with_new['f1']}")

    # ---- Step 2: PCA squashes the columns into fewer ones. Does it lose anything? ----
    print("\nStep 2: PCA (keep 95% of the information)")
    columns_before = make_preprocessor(X_train).fit_transform(X_train).shape[1]
    pca_pipeline = Pipeline([
        ("prepare", make_preprocessor(X_train)),
        ("pca", PCA(n_components=0.95)),
        ("model", new_logistic()),
    ])
    pca = cv_scores(pca_pipeline, X_train, y_train)
    pca_pipeline.fit(X_train, y_train)
    columns_after = pca_pipeline.named_steps["pca"].n_components_
    no_pca = feature_check["Logistic Regression"]["with_new_features"]
    print(f"  columns: {columns_before} -> {columns_after}")
    print("  Logistic Regression without PCA:", no_pca)
    print("  Logistic Regression with PCA:   ", pca)
    print("  -> PCA keeps almost all of the score with far fewer columns, but the new columns are blends that"
          if pca["f1"] >= no_pca["f1"] - 0.005 else "  -> PCA loses signal here, so we do not use it")
    if pca["f1"] >= no_pca["f1"] - 0.005:
        print("     nobody can explain, so we note it but do not use it.")

    # ---- Step 3: ensembles. All of them use the new features. ----
    print("\nStep 3: bagging and boosting models")
    # Churners are about 1 stayer for every 0.36 churners; this number tells XGBoost to count a churner more.
    churn_weight = round(float((y_train == 0).sum() / (y_train == 1).sum()), 2)
    models = {
        # --- bagging family: many trees trained separately, then they vote ---
        "Random Forest": new_forest(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                            class_weight="balanced", random_state=RANDOM_STATE),
        "Bagging (plain trees)": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6, class_weight="balanced"),
            n_estimators=100, random_state=RANDOM_STATE),
        # --- boosting family: trees added one by one, each fixing the mistakes of the ones before ---
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
            n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, scale_pos_weight=churn_weight,
            eval_metric="logloss", random_state=RANDOM_STATE)
    else:
        print("  (XGBoost is not available here, so it is skipped. Install it with: pip install xgboost)")

    pipelines = {}
    validation = {}
    for name, model in models.items():
        print("  training", name, "...")
        pipelines[name] = make_pipeline(X_train, model)
        validation[name] = cv_scores(pipelines[name], X_train, y_train)
        pipelines[name].fit(X_train, y_train)

    # The best model is chosen by validation F1 (on the training customers, not the test ones).
    champion = None
    for name in models:
        if champion is None or validation[name]["f1"] > validation[champion]["f1"]:
            champion = name

    # ---- Step 4: test on the test customers (once) and compare with the Day 3 champion ----
    print("\nStep 4: test customers")
    day3_model = joblib.load(DAY3_MODEL_FILE)
    test_scores = {"Day 3 champion (old columns)": evaluate(day3_model, X_test[old_columns], y_test)}
    for name in models:
        test_scores[name] = evaluate(pipelines[name], X_test, y_test)

    print(f"  {'Model':<30}{'Valid F1':>9}{'Recall':>8}{'Precision':>11}{'F1':>8}{'ROC-AUC':>9}")
    for name, s in test_scores.items():
        valid = validation[name]["f1"] if name in validation else "-"
        mark = "  <- final model" if name == champion else ""
        print(f"  {name:<30}{valid:>9}{s['recall']:>8}{s['precision']:>11}{s['f1']:>8}{s['roc_auc']:>9}{mark}")

    day3 = test_scores["Day 3 champion (old columns)"]
    final = test_scores[champion]
    if final["f1"] > day3["f1"]:
        verdict = f"{champion} beats the Day 3 champion (F1 {final['f1']} vs {day3['f1']})."
    else:
        verdict = (f"{champion} does not beat the Day 3 champion (F1 {final['f1']} vs {day3['f1']}); "
                   "compare recall and ROC-AUC too.")
    print("  ->", verdict)

    # ---- Step 5: lock the final model ----
    # The saved pipeline contains the data preparation (scaling + one-hot) AND the model,
    # so serving applies exactly the same preparation. The new columns come from src/features.py.
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(pipelines[champion], DAY4_MODEL_FILE)
    with open(DAY4_RESULTS_FILE, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "new_features": NEW_COLUMNS,
            "feature_check": feature_check,
            "pca": {"columns_before": int(columns_before), "columns_after": int(columns_after),
                    "scores": pca, "scores_without_pca": no_pca},
            "validation_scores": validation,
            "final_model": champion,
            "verdict": verdict,
            "test_scores": test_scores,
        }, f, indent=2)
    print("\nSaved final model ->", DAY4_MODEL_FILE)
    print("Saved results     ->", DAY4_RESULTS_FILE)


if __name__ == "__main__":
    main()
