"""Day 2: train Decision Tree, Random Forest and SVM, compare them, keep the best.

Run the Day 1 script first (it makes the baseline model), then from the main project folder:
    python -m src.train_day2
"""
import json
import os
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (CHAMPION_MODEL_FILE, DAY1_MODEL_FILE, DAY2_COMPARISON_FILE,
                        MODEL_FOLDER, RANDOM_STATE)
from src.data import get_train_test_data
from src.evaluate import evaluate
from src.pipeline import make_pipeline


def main():
    # Same customers and same preparation as Day 1, so the comparison is fair.
    X_train, X_test, y_train, y_test = get_train_test_data()

    # The three new models. Every one uses the same random seed.
    # class_weight="balanced" tells a model that a churner matters more than a
    # stayer. Only about 1 in 4 customers churns, so without it the models tend
    # to ignore the churners we most want to catch.
    new_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }

    results = {}              # model name -> scores
    trained_pipelines = {}    # model name -> trained pipeline

    # The Day 1 model is loaded (not retrained) so we can compare against it.
    print("Testing the Day 1 model (logistic regression)...")
    day1_pipeline = joblib.load(DAY1_MODEL_FILE)
    results["Logistic Regression (Day 1)"] = evaluate(day1_pipeline, X_test, y_test)

    for name, model in new_models.items():
        print("Training", name, "...")
        pipeline = make_pipeline(X_train, model)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        results[name] = evaluate(pipeline, X_test, y_test)

    # Pick the champion: the new model that catches the most churners (highest recall).
    # If two models tie on recall, the one with the better F1 wins.
    champion = None
    for name in new_models:
        if champion is None:
            champion = name
        elif (results[name]["recall"], results[name]["f1"]) > (results[champion]["recall"], results[champion]["f1"]):
            champion = name

    # ---- show the comparison table ----
    print()
    print(f"{'Model':<28}{'Recall':>8}{'Precision':>11}{'F1':>8}{'ROC-AUC':>9}{'Accuracy':>10}")
    print("-" * 74)
    for name, s in results.items():
        mark = "  <- champion" if name == champion else ""
        print(f"{name:<28}{s['recall']:>8}{s['precision']:>11}{s['f1']:>8}{s['roc_auc']:>9}{s['accuracy']:>10}{mark}")

    day1 = results["Logistic Regression (Day 1)"]
    best = results[champion]
    why = (f"{champion} won because it caught the most churners: recall {best['recall']} "
           f"against {day1['recall']} for the Day 1 model. "
           f"The cost is precision ({best['precision']} vs {day1['precision']}), "
           f"so more loyal customers are flagged by mistake.")
    print()
    print(why)

    # ---- save the champion next to the Day 1 model, plus the comparison ----
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(trained_pipelines[champion], CHAMPION_MODEL_FILE)

    comparison = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "test_rows": len(X_test),
        "champion": champion,
        "why": why,
        "scores": results,
    }
    with open(DAY2_COMPARISON_FILE, "w") as f:
        json.dump(comparison, f, indent=2)

    print("Saved champion   ->", CHAMPION_MODEL_FILE)
    print("Saved comparison ->", DAY2_COMPARISON_FILE)


if __name__ == "__main__":
    main()
