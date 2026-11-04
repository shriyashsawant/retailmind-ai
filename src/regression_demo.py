"""Regression demo: predict a NUMBER (the monthly bill) instead of a yes/no answer.

Classification answers "which group?" (will this customer leave: yes or no).
Regression answers "how much?" (what will this customer's monthly bill be: 79.30).
Linear Regression is the simplest regression model. Logistic Regression, despite its name,
is the classification partner we use for churn.

Run from the main project folder:   python -m src.regression_demo
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.data import clean_data, load_data
from src.pipeline import make_pipeline


def main():
    df = clean_data(load_data())

    # The answer we predict is now a number: MonthlyCharges.
    # We drop TotalCharges too, because it is roughly MonthlyCharges x tenure (that would be cheating).
    y = df["MonthlyCharges"]
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN, "MonthlyCharges", "TotalCharges"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),     # linear regression with a brake on very large weights
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    }

    print(f"Predicting MonthlyCharges (average bill is {y.mean():.2f})")
    print(f"{'Model':<26}{'Average error':>15}{'R2':>8}")
    for name, model in models.items():
        pipeline = make_pipeline(X_train, model)
        pipeline.fit(X_train, y_train)
        guesses = pipeline.predict(X_test)
        error = mean_absolute_error(y_test, guesses)   # on average the guess is this far from the real bill
        r2 = r2_score(y_test, guesses)                 # 1.0 = perfect, 0.0 = no better than guessing the average
        print(f"{name:<26}{error:>15.2f}{r2:>8.3f}")


if __name__ == "__main__":
    main()
