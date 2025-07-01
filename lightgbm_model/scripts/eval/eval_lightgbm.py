# eval_model.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm_model.scripts.config_lightgbm import RESULTS_DIR, MODEL_DIR, DATA_PATH
from lightgbm_model.scripts.utils import load_lightgbm_model
from joblib import load

# === Ergebnisse-Ordner vorbereiten ===
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Modell und eval_result laden ===
# Modell laden
model = load_lightgbm_model()

# Eval laden
with open(os.path.join(RESULTS_DIR, "lightgbm_eval_result.pkl"), "rb") as f:
    eval_result = pickle.load(f)
X_train = pd.read_csv(os.path.join(RESULTS_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(RESULTS_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(RESULTS_DIR, "y_test.csv"))

# === Lernkurve ===
train_rmse = eval_result['training']['rmse']
valid_rmse = eval_result['valid_1']['rmse']

plt.figure(figsize=(10, 5))
plt.plot(train_rmse, label='Train RMSE')
plt.plot(valid_rmse, label='Valid RMSE')
plt.axvline(model.best_iteration_, color='gray', linestyle='--', label='Best Iteration')
plt.xlabel("Boosting Round")
plt.ylabel("RMSE")
plt.title("LightGBM Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lightgbm_learning_curve.png"))
#plt.show()

# === Metriken berechnen ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test.values.flatten() - y_pred) / np.where(y_test.values.flatten() == 0, 1e-10, y_test.values.flatten()))) * 100
r2 = r2_score(y_test, y_pred)

print(f"Test MAPE: {mape:.5f} %")
print(f"Test MAE: {mae:.5f}")
print(f"Test RMSE: {rmse:.5f}")
print(f"Test R2: {r2:.5f}")

metrics = {
    "model": "LightGBM",
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "MAPE (%)": round(mape, 2),
    "R2": round(r2, 4),
    "unit": "MW"
}

# Pfad setzen
output_path = os.path.join(RESULTS_DIR, "evaluation_metrics_lightgbm.json")
# Speichern
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metriken gespeichert unter {output_path}")

# === Feature Importance ===
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Feature Importance")
plt.title("LightGBM Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lightgbm_feature_importance.png"))
#plt.show()

# === Vergleichsplots ===
results_df = pd.DataFrame({
    "True Consumption (MW)": y_test.values.flatten(),
    "Predicted Consumption (MW)": y_pred
})

# Timestamps anhängen
full_df = pd.read_csv(DATA_PATH)
test_dates = full_df.iloc[int(len(full_df)*0.8):]["date"].reset_index(drop=True)
results_df["Timestamp"] = pd.to_datetime(test_dates)

# Voller Plot
plt.figure(figsize=(15, 6))
plt.plot(results_df["Timestamp"], results_df["True Consumption (MW)"], label="True", color="darkblue")
plt.plot(results_df["Timestamp"], results_df["Predicted Consumption (MW)"], label="Predicted", color="red", linestyle="--")
plt.title("Predicted vs True Consumption")
plt.xlabel("Timestamp")
plt.ylabel("Consumption (MW)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lightgbm_comparison_plot.png"))
#plt.show()

# Subset Plot
subset = results_df.iloc[:len(results_df) // 10]
plt.figure(figsize=(15, 6))
plt.plot(subset["Timestamp"], subset["True Consumption (MW)"], label="True", color="darkblue")
plt.plot(subset["Timestamp"], subset["Predicted Consumption (MW)"], label="Predicted", color="red", linestyle="--")
plt.title("Predicted vs True (First decile)")
plt.xlabel("Timestamp")
plt.ylabel("Consumption (MW)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lightgbm_prediction_with_timestamp.png"))
#plt.show()


# === Ens message ===
print("\nEvaluation completed.")
print(f"All Plots stored in:\n→ {RESULTS_DIR}")
