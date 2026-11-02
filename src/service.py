import argparse
import json

import joblib
import pandas as pd

from .config import DECISION_THRESHOLD, MODEL_PATH


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def _to_frame(records):
    if isinstance(records, dict):
        return pd.DataFrame([records])
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame(records)


def predict(records, threshold=DECISION_THRESHOLD, model_path=MODEL_PATH):
    """Score one customer (dict) or many (list[dict] / DataFrame).

    Returns a list of dicts with the churn probability and a yes/no prediction.
    """
    artifact = load_model(model_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    frame = _to_frame(records)

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    frame = frame[feature_columns].copy()
    frame["TotalCharges"] = pd.to_numeric(
        frame["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    probabilities = pipeline.predict_proba(frame)[:, 1]

    results = []
    for probability in probabilities:
        probability = float(probability)
        if probability >= 0.7:
            risk_tier = "high"
        elif probability >= 0.4:
            risk_tier = "medium"
        else:
            risk_tier = "low"
        results.append(
            {
                "churn_probability": round(probability, 4),
                "churn_prediction": "Yes" if probability >= threshold else "No",
                "risk_tier": risk_tier,
            }
        )
    return results


SAMPLE_CUSTOMERS = [
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    },
    {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 105.5,
        "TotalCharges": 7174.0,
    },
]


def main():
    parser = argparse.ArgumentParser(description="RetailMind churn prediction service")
    parser.add_argument(
        "--input",
        help="Path to a CSV file of customers. Defaults to built-in demo customers.",
    )
    args = parser.parse_args()

    if args.input:
        records = pd.read_csv(args.input)
    else:
        records = SAMPLE_CUSTOMERS

    for index, result in enumerate(predict(records), start=1):
        print(json.dumps({"customer": index, **result}))


if __name__ == "__main__":
    main()
