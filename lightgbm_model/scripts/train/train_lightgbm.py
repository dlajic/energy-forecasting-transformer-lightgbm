# train_lightgbm.py

import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, record_evaluation

from lightgbm_model.scripts.config_lightgbm import (
    DATA_PATH,
    FEATURES,
    TARGET,
    LIGHTGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    RESULTS_DIR,
    MODEL_DIR
)

# === Load Data ===
df = pd.read_csv(DATA_PATH)

# Drop date (used later for plots only)
df = df.drop(columns=["date"], errors="ignore")

# === Time-based Split (70% train, 10% valid, 20% test) ===
train_size = int(len(df) * 0.7)
valid_size = int(len(df) * 0.1)
df_train = df.iloc[:train_size]
df_valid = df.iloc[train_size:train_size + valid_size]
df_test = df.iloc[train_size + valid_size:]

X_train, y_train = df_train[FEATURES], df_train[TARGET]
X_valid, y_valid = df_valid[FEATURES], df_valid[TARGET]
X_test, y_test = df_test[FEATURES], df_test[TARGET]


# === Init LightGBM model ===
eval_result = {}

model = LGBMRegressor(
    **LIGHTGBM_PARAMS,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[
        early_stopping(EARLY_STOPPING_ROUNDS),
        record_evaluation(eval_result)
    ]
)

# === Save model ===
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "lightgbm_final_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

# === Save evaluation results ===
os.makedirs(RESULTS_DIR, exist_ok=True)
eval_result_path = os.path.join(RESULTS_DIR, "lightgbm_eval_result.pkl")

with open(eval_result_path, "wb") as f:
    pickle.dump(eval_result, f)

print(f"Model saved to: {model_path}")
print(f"Eval results saved to: {eval_result_path}")

# === Save data for evaluation ===
X_train.to_csv(os.path.join(RESULTS_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(RESULTS_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(RESULTS_DIR, "y_test.csv"), index=False)

