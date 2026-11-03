# Settings used by every script in this project.
# Change a value here and it changes everywhere.
# Run all commands from the main project folder (the one that contains "src").

# --- files ---
DATA_FILE = "data/raw/Telco-Customer-Churn.csv"
REPORT_FOLDER = "reports"
MODEL_FOLDER = "models"

DAY1_MODEL_FILE = "models/churn_pipeline.joblib"         # Day 1: logistic regression
DAY1_METRICS_FILE = "models/metrics.json"
CHAMPION_MODEL_FILE = "models/champion_pipeline.joblib"  # Day 2: the best of the three new models
DAY2_COMPARISON_FILE = "models/day2_comparison.json"

# --- column names ---
TARGET_COLUMN = "Churn"          # what we want to predict (Yes / No)
ID_COLUMN = "customerID"         # just a label, not useful for learning
NUMBER_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# --- training settings ---
TEST_SIZE = 0.2          # 20% of customers are kept aside to test the model
RANDOM_STATE = 42        # a fixed number so every run gives the same answer
THRESHOLD = 0.5          # churn probability at or above this means "will churn"
