"""Day 4: new features, PCA, and stronger models (bagging + boosting). Then lock the final model.

Run Day 3 first (it makes the model we compare against), then from the main project folder:
    python -m src.train_day4
"""
import json
import os
from datetime import datetime

import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.config import DAY3_MODEL_FILE, DAY4_MODEL_FILE, DAY4_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE
from src.data import get_train_test_data
from src.evaluate import evaluate
from src.features import NEW_COLUMNS
from src.pipeline import make_pipeline, make_preprocessor


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

    # ---- Step 1: do the new features help? (same model, with and without them) ----
    print("Step 1: do the new features help?")
    without = cv_scores(make_pipeline(X_train[old_columns], new_logistic()), X_train[old_columns], y_train)
    with_new = cv_scores(make_pipeline(X_train, new_logistic()), X_train, y_train)
    print("  without new features:", without)
    print("  with new features:   ", with_new)
    features_help = with_new["f1"] > without["f1"]
    print("  ->", "they help" if features_help else "they do not help much")

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
    print(f"  columns: {columns_before} -> {columns_after}")
    print("  without PCA:", with_new)
    print("  with PCA:   ", pca)
    use_pca = pca["f1"] >= with_new["f1"] - 0.005
    print("  ->", "PCA keeps the score with fewer columns, but it makes the model harder to explain, so we only note it"
          if use_pca else "PCA loses signal here, so we do not use it")

    # ---- Step 3: stronger models, both using the new features ----
    print("\nStep 3: bagging and boosting")
    models = {
        # bagging: many trees, each trained on a random part of the data, then they vote
        "Random Forest (bagging)": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE),
        # boosting: trees are added one by one, each one fixing the mistakes of the ones before.
        # (XGBoost does the same job; scikit-learn's version needs no extra install.)
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }
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

    print(f"  {'Model':<30}{'Recall':>8}{'Precision':>11}{'F1':>8}{'ROC-AUC':>9}")
    for name, s in test_scores.items():
        mark = "  <- final model" if name == champion else ""
        print(f"  {name:<30}{s['recall']:>8}{s['precision']:>11}{s['f1']:>8}{s['roc_auc']:>9}{mark}")

    day3 = test_scores["Day 3 champion (old columns)"]
    final = test_scores[champion]
    if final["f1"] > day3["f1"]:
        verdict = f"{champion} beats the Day 3 champion (F1 {final['f1']} vs {day3['f1']})."
    else:
        verdict = (f"{champion} does not beat the Day 3 champion (F1 {final['f1']} vs {day3['f1']}); "
                   "we still keep the Day 4 model because it also uses the new features, so compare recall and ROC-AUC too.")
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
            "feature_check": {"without_new_features": without, "with_new_features": with_new, "they_help": features_help},
            "pca": {"columns_before": int(columns_before), "columns_after": int(columns_after),
                    "scores": pca, "scores_without_pca": with_new},
            "validation_scores": validation,
            "final_model": champion,
            "verdict": verdict,
            "test_scores": test_scores,
        }, f, indent=2)
    print("\nSaved final model ->", DAY4_MODEL_FILE)
    print("Saved results     ->", DAY4_RESULTS_FILE)


if __name__ == "__main__":
    main()
