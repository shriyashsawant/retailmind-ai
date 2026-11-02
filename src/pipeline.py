from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor(feature_columns):
    """Encode categoricals and scale numerics in one reusable transformer."""
    numeric = [c for c in feature_columns if c in NUMERIC_FEATURES]
    categorical = [c for c in feature_columns if c not in numeric]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(feature_columns):
    """Day 1 baseline: preprocessing + logistic regression, end to end."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_columns)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
