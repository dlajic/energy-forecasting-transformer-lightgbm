# config.py
import os

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "energy_consumption_aggregated_cleaned.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# === Feature-Definition ===
FEATURES = [
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "rolling_mean_6h",
    "month_sin", "month_cos",
    "temperature_c",
    "consumption_last_week",
    "consumption_yesterday",
    "consumption_last_hour"
]
TARGET = "consumption_MW"

# === Hyperparameters fpr LightGBM ===
LIGHTGBM_PARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 15,
    'max_depth': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 0.0,
    'min_split_gain': 0.0,
    'n_estimators': 1000,
    'objective': 'regression'}

# === Early Stopping ===
EARLY_STOPPING_ROUNDS = 50

