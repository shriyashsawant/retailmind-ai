from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORTS_DIR = ROOT_DIR / "reports"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

TEST_SIZE = 0.2
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
