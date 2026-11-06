"""Score customers with the Day 5 neural network.

Uses the same sample customers and the same risk words as src/service.py,
so the ANN and the classical model can be compared side by side.

Run from the main project folder (after src.train_day5):   python -m src.ann_service
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import keras
import pandas as pd

from src.config import ANN_MODEL_FILE, ANN_PREPARE_FILE, THRESHOLD
from src.service import SAMPLE_CUSTOMERS, risk_label


def load_ann():
    """Load the saved network and the saved data-preparation step."""
    return keras.models.load_model(ANN_MODEL_FILE), joblib.load(ANN_PREPARE_FILE)


def predict_with_ann(customers):
    """customers is a list of dictionaries (one per customer). Returns one answer per customer."""
    network, prepare = load_ann()
    table = pd.DataFrame(customers)
    ready = prepare.transform(table).astype("float32")      # same scaling + one-hot as in training
    probabilities = network.predict(ready, verbose=0).ravel()

    answers = []
    for number, probability in enumerate(probabilities, start=1):
        answers.append({
            "customer": number,
            "churn_probability": round(float(probability), 4),
            "churn_prediction": "Yes" if probability >= THRESHOLD else "No",
            "risk_tier": risk_label(float(probability)),
        })
    return answers


if __name__ == "__main__":
    for answer in predict_with_ann(SAMPLE_CUSTOMERS):
        print(json.dumps(answer))
