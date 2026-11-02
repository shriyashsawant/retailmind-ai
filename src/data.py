import json

import pandas as pd

from .config import ID_COLUMN, RAW_DATA_PATH, REPORTS_DIR, TARGET_COLUMN


def load_raw(path=RAW_DATA_PATH):
    """Load the Telco Customer Churn dataset."""
    return pd.read_csv(path)


def clean(df):
    """Normalise dtypes and the target without leaking test information."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    return df


def split_features_target(df):
    """Return (X, y), dropping the customer identifier from the features."""
    X = df.drop(columns=[c for c in (ID_COLUMN, TARGET_COLUMN) if c in df.columns])
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    return X, y


def profile(df):
    """Build a small, JSON-serialisable profile of a dataframe."""
    target_counts = {}
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        total = int(counts.sum())
        target_counts = {
            str(label): {
                "count": int(count),
                "share": round(float(count) / total, 4),
            }
            for label, count in counts.items()
        }

    missing = df.isna().sum()
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            blanks = df[column].astype(str).str.strip().eq("").sum()
            missing[column] += int(blanks)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "schema": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "missing_values": {name: int(count) for name, count in missing.items() if count},
        "target_balance": target_counts,
        "numeric_summary": {
            column: {stat: round(float(value), 4) for stat, value in stats.items()}
            for column, stats in df.select_dtypes("number").describe().to_dict().items()
        },
    }


def save_profile(df, path=None):
    path = path or REPORTS_DIR / "data_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(profile(df), handle, indent=2)
    return path
