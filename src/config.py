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
DAY3_MODEL_FILE = "models/churn_v3.joblib"               # Day 3: the tuned champion
DAY3_RESULTS_FILE = "models/day3_tuning.json"
DAY4_MODEL_FILE = "models/churn_final.joblib"            # Day 4: the final production model
DAY4_RESULTS_FILE = "models/day4_results.json"
BEST_MODEL_FILE = "models/churn_best_overall.joblib"     # the winner of the full fair comparison
ALL_RESULTS_FILE = "models/all_models_results.json"
ANN_MODEL_FILE = "models/churn_ann.keras"                # Day 5: the Keras neural network
ANN_PREPARE_FILE = "models/churn_ann_prepare.joblib"     # Day 5: the scaling + one-hot step the ANN needs
DAY5_RESULTS_FILE = "models/day5_results.json"
CNN_MODEL_FILE = "models/catalog_cnn.keras"              # Day 6: the catalog image classifier (Fashion-MNIST)
PRODUCT_CNN_MODEL_FILE = "models/catalog_cnn_products.keras"   # Day 6: same idea on real product photos (if the data is present)
DAY6_RESULTS_FILE = "models/day6_results.json"
RESNET_MODEL_FILE = "models/catalog_resnet50.keras"      # Day 7: transfer-learning classifier (large, not committed)
DAY7_RESULTS_FILE = "models/day7_results.json"
DETECTIONS_FILE = "reports/day7_detections.json"         # Day 7: what YOLO found in the sample image
DETECTIONS_IMAGE = "reports/day7_detections.jpg"         # Day 7: the sample image with boxes drawn on it
PRODUCT_IMAGES_FOLDER ="data/fashion_product_images"    # Kaggle: paramaggarwal/fashion-product-images-small (images/ + styles.csv)

# --- column names ---
TARGET_COLUMN = "Churn"          # what we want to predict (Yes / No)
ID_COLUMN = "customerID"         # just a label, not useful for learning
NUMBER_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# --- training settings ---
TEST_SIZE = 0.2          # 20% of customers are kept aside to test the model
RANDOM_STATE = 42        # a fixed number so every run gives the same answer
THRESHOLD = 0.5          # churn probability at or above this means "will churn"
