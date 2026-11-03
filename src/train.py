"""Day 1: train the baseline model (logistic regression).

Run from the main project folder:   python -m src.train
"""
import json
import os
from datetime import datetime

import joblib
from sklearn.linear_model import LogisticRegression

from src.config import DAY1_METRICS_FILE, DAY1_MODEL_FILE, MODEL_FOLDER, RANDOM_STATE
from src.data import get_train_test_data, save_data_profile
from src.evaluate import evaluate
from src.pipeline import make_pipeline


def main():
    print("1. Looking at the data...")
    profile = save_data_profile()
    print("   customers:", profile["rows"], "| churn rate:", profile["churn_rate"])

    print("2. Splitting into training and test customers...")
    X_train, X_test, y_train, y_test = get_train_test_data()

    print("3. Training logistic regression on", len(X_train), "customers...")
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    pipeline = make_pipeline(X_train, model)
    pipeline.fit(X_train, y_train)

    print("4. Testing on", len(X_test), "customers the model has never seen...")
    scores = evaluate(pipeline, X_test, y_test)

    print("5. Saving the model and its scores...")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(pipeline, DAY1_MODEL_FILE)

    info = {
        "model": "LogisticRegression",
        "dataset": "Telco Customer Churn",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": scores,
    }
    with open(DAY1_METRICS_FILE, "w") as f:
        json.dump(info, f, indent=2)

    print()
    print("Day 1 scores")
    print("------------")
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{name:>10}: {scores[name]}")
    print("Saved model  ->", DAY1_MODEL_FILE)
    print("Saved scores ->", DAY1_METRICS_FILE)


if __name__ == "__main__":
    main()
