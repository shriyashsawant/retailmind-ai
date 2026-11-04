import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (DATA_FILE, ID_COLUMN, RANDOM_STATE, REPORT_FOLDER,
                        TARGET_COLUMN, TEST_SIZE)
from src.features import add_features


def load_data():
    """Read the customer table from the CSV file."""
    return pd.read_csv(DATA_FILE)


def clean_data(df):
    """Fix the two columns that are not ready to use."""
    df = df.copy()  # work on a copy so the original table is not changed

    # TotalCharges is stored as text, and 11 brand-new customers have a blank.
    # Turn it into a number; a blank becomes 0 (they have not been billed yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Churn is "Yes" / "No". Models need numbers, so Yes -> 1 and No -> 0.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return df


def get_train_test_data(new_features=False):
    """Load, clean and split the data. Every script uses this, so all models
    are trained and tested on exactly the same customers.

    new_features=True adds the Day 4 columns from src/features.py.
    """
    df = clean_data(load_data())

    y = df[TARGET_COLUMN]                                   # the answer we want to predict
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])         # everything else is a clue
    if new_features:
        X = add_features(X)

    # stratify=y keeps the same share of churners in the train and test parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test


def save_data_profile():
    """Write a few simple facts about the raw data to reports/data_profile.json."""
    raw = load_data()
    blank_total_charges = int(raw["TotalCharges"].astype(str).str.strip().eq("").sum())
    churn_count = int((raw[TARGET_COLUMN] == "Yes").sum())

    profile = {
        "rows": len(raw),
        "columns": raw.shape[1],
        "churned_customers": churn_count,
        "churn_rate": round(churn_count / len(raw), 4),
        "blank_total_charges": blank_total_charges,
    }

    os.makedirs(REPORT_FOLDER, exist_ok=True)
    file_path = os.path.join(REPORT_FOLDER, "data_profile.json")
    with open(file_path, "w") as f:
        json.dump(profile, f, indent=2)
    return profile
