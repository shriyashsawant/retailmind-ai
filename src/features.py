"""Day 4: make new columns ("features") that help the model spot churners.

Each new column is worked out from the SAME customer's own row, never from other
customers, so nothing from the test customers leaks into training.
"""
import pandas as pd

# The names of the 6 columns that add_features() creates.
NEW_COLUMNS = ["tenure_group", "charge_ratio", "is_month_to_month",
               "has_internet", "number_of_services", "pays_automatically"]

# The extra services a customer can switch on.
SERVICE_COLUMNS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(X):
    """Return a copy of the customer table with 6 new columns added."""
    X = X.copy()

    # 1. Tenure bucket: very new customers leave far more often than long-time ones.
    X["tenure_group"] = pd.cut(
        X["tenure"], bins=[-1, 12, 24, 48, 100],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"]).astype(str)

    # 2. Charge ratio: how today's monthly price compares to what they paid on average.
    #    A big number means the price went up. (+1 stops us dividing by zero for new customers.)
    average_charge = X["TotalCharges"] / (X["tenure"] + 1)
    X["charge_ratio"] = X["MonthlyCharges"] / (average_charge + 1)

    # 3. Contract flag: month-to-month customers can leave at any time.
    X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)

    # 4. Service flags: does the customer have internet, how many extras do they use,
    #    and do they pay automatically? (Customers who are more "tied in" tend to stay.)
    X["has_internet"] = (X["InternetService"] != "No").astype(int)
    X["number_of_services"] = (X[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    X["pays_automatically"] = X["PaymentMethod"].str.contains("automatic").astype(int)

    return X
