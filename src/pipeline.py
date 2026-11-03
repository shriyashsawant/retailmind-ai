from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMBER_COLUMNS


def make_preprocessor(X):
    """Get the data ready for any model.

    - number columns are scaled so big numbers do not drown out small ones
    - text columns (like "Contract") are turned into 0/1 columns (one-hot)
    """
    text_columns = []
    for column in X.columns:
        if column not in NUMBER_COLUMNS:
            text_columns.append(column)

    return ColumnTransformer([
        ("numbers", StandardScaler(), NUMBER_COLUMNS),
        ("text", OneHotEncoder(handle_unknown="ignore"), text_columns),
    ])


def make_pipeline(X, model):
    """Join the preparation step and a model into one object.

    Every model (Day 1 and Day 2) goes through this same function, so they all
    learn from exactly the same prepared features. That makes the comparison fair.
    """
    return Pipeline([
        ("prepare", make_preprocessor(X)),
        ("model", model),
    ])
