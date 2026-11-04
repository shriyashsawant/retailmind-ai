import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X):
    """Get the data ready for any model.

    - number columns are scaled so big numbers do not drown out small ones
    - text columns (like "Contract") are turned into 0/1 columns (one-hot)

    The columns are sorted by type automatically, so new columns that we add
    later (Day 4) are handled without changing this code.
    """
    number_columns = []
    text_columns = []
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            number_columns.append(column)
        else:
            text_columns.append(column)

    return ColumnTransformer([
        ("numbers", StandardScaler(), number_columns),
        ("text", OneHotEncoder(handle_unknown="ignore", sparse_output=False), text_columns),
    ])


def make_pipeline(X, model, use_pca=False):
    """Join the preparation step and a model into one object.

    Every model goes through this same function, so they all learn from exactly
    the same prepared features. That makes the comparison fair.

    use_pca=True squashes the prepared columns into fewer ones (keeping 95% of
    the information) before the model sees them.
    """
    steps = [("prepare", make_preprocessor(X))]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95)))
    steps.append(("model", model))
    return Pipeline(steps)
