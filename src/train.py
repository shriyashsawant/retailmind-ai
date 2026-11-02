import json
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split

from .config import (
    DECISION_THRESHOLD,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data import clean, load_raw, save_profile, split_features_target
from .evaluate import evaluate_model
from .pipeline import build_pipeline


def main():
    print("Loading raw data...")
    raw = load_raw()
    save_profile(raw)
    print(f"  rows={len(raw):,} columns={raw.shape[1]}")

    print("Cleaning data...")
    df = clean(raw)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training baseline on {len(X_train):,} rows...")
    pipeline = build_pipeline(feature_columns=list(X.columns))
    pipeline.fit(X_train, y_train)

    print("Evaluating on hold-out set...")
    metrics, _, _ = evaluate_model(
        pipeline, X_test, y_test, threshold=DECISION_THRESHOLD
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": list(X.columns),
        "target_column": "Churn",
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    joblib.dump(artifact, MODEL_PATH)

    summary = dict(metrics)
    summary.pop("classification_report")

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "LogisticRegression",
                "dataset": "Telco Customer Churn",
                "trained_at": artifact["trained_at"],
                "train_rows": artifact["train_rows"],
                "test_rows": artifact["test_rows"],
                "metrics": summary,
            },
            handle,
            indent=2,
        )

    print("\nBaseline metrics")
    print("----------------")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key:>10}: {metrics[key]:.4f}")
    print("\nClassification report")
    print(metrics["classification_report"])
    print(f"Saved pipeline -> {MODEL_PATH}")
    print(f"Saved metrics  -> {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    main()
