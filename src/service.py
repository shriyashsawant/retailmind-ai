"""Predict churn for new customers.

From the main project folder:
    python -m src.service                      # try two example customers
    python -m src.service my_customers.csv     # score customers from a CSV file
"""
import json
import os
import sys

import joblib
import pandas as pd

from src.config import (BEST_MODEL_FILE, CHAMPION_MODEL_FILE, DAY1_MODEL_FILE,
                        DAY3_MODEL_FILE, DAY4_MODEL_FILE, THRESHOLD)
from src.features import add_features

# Two made-up customers so you can try the service straight away.
SAMPLE_CUSTOMERS = [
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    },
    {
        "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 68, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 105.5, "TotalCharges": 7174.0,
    },
]


def load_model(model_file=None):
    """Load a saved model. By default use the best one that exists:
    the winner of the full comparison, then Day 4, then Day 3, then Day 2, then the Day 1 baseline."""
    if model_file is None:
        for candidate in [BEST_MODEL_FILE, DAY4_MODEL_FILE, DAY3_MODEL_FILE, CHAMPION_MODEL_FILE, DAY1_MODEL_FILE]:
            if os.path.exists(candidate):
                model_file = candidate
                break
    return joblib.load(model_file)


def risk_label(probability):
    """Turn a churn probability into an easy word."""
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    else:
        return "low"


def predict(customers, model_file=None):
    """Score one customer (a dict), a list of dicts, or a table (DataFrame).

    Returns one result per customer.
    """
    pipeline = load_model(model_file)

    # Put whatever we were given into a table with one row per customer.
    if isinstance(customers, dict):
        table = pd.DataFrame([customers])
    else:
        table = pd.DataFrame(customers)

    # Same fix as in training: TotalCharges must be a number, blank becomes 0.
    table["TotalCharges"] = pd.to_numeric(table["TotalCharges"], errors="coerce").fillna(0)

    # Build the Day 4 columns too. Older models simply ignore the ones they do not use.
    table = add_features(table)

    # The model remembers which columns it was trained on.
    needed_columns = list(pipeline.feature_names_in_)
    missing = [c for c in needed_columns if c not in table.columns]
    if missing:
        raise ValueError("These columns are missing: " + ", ".join(missing))
    table = table[needed_columns].copy()

    probabilities = pipeline.predict_proba(table)[:, 1]

    results = []
    for probability in probabilities:
        probability = float(probability)
        if probability >= THRESHOLD:
            answer = "Yes"
        else:
            answer = "No"
        results.append({
            "churn_probability": round(probability, 4),
            "churn_prediction": answer,
            "risk_tier": risk_label(probability),
        })
    return results


def main():
    if len(sys.argv) > 1:
        customers = pd.read_csv(sys.argv[1])   # the CSV path given on the command line
    else:
        customers = SAMPLE_CUSTOMERS

    number = 1
    for result in predict(customers):
        print(json.dumps({"customer": number, **result}))
        number = number + 1


if __name__ == "__main__":
    main()
