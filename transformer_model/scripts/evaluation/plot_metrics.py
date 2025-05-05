# plot_metrics.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformer_model.scripts.config_transformer import RESULTS_DIR

# === Plot 1: Training Metrics ===

# Load training metrics
training_metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")
with open(training_metrics_path, "r") as f:
    metrics = json.load(f)

train_losses = metrics["train_losses"]
test_mses = metrics["test_mses"]
test_maes = metrics["test_maes"]

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, len(test_mses) + 1), test_mses, label="Test MSE", color="red")
plt.plot(range(1, len(test_maes) + 1), test_maes, label="Test MAE", color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss / Metric")
plt.title("Training Loss vs Test Metrics")
plt.legend()
plt.grid(True)

plot_path = os.path.join(RESULTS_DIR, "training_plot.png")
plt.savefig(plot_path)
print(f"[Saved] Training metrics plot: {plot_path}")
plt.show()


# === Plot 2: Predictions vs Ground Truth (Full Range) ===

# Load comparison results
comparison_path = os.path.join(RESULTS_DIR, "test_results.csv")
df_comparison = pd.read_csv(comparison_path, parse_dates=["Timestamp"])

plt.figure(figsize=(15, 6))
plt.plot(df_comparison["Timestamp"], df_comparison["True Consumption (MW)"], label="True", color="darkblue")
plt.plot(df_comparison["Timestamp"], df_comparison["Predicted Consumption (MW)"], label="Predicted", color="red", linestyle="--")
plt.title("Energy Consumption: Predictions vs Ground Truth")
plt.xlabel("Time")
plt.ylabel("Consumption (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "comparison_plot_full.png")
plt.savefig(plot_path)
print(f"[Saved] Full range comparison plot: {plot_path}")
plt.show()


# === Plot 3: Predictions vs Ground Truth (First Month) ===

first_month_start = df_comparison["Timestamp"].min()
first_month_end = first_month_start + pd.Timedelta(days=25)
df_first_month = df_comparison[(df_comparison["Timestamp"] >= first_month_start) & (df_comparison["Timestamp"] <= first_month_end)]

plt.figure(figsize=(15, 6))
plt.plot(df_first_month["Timestamp"], df_first_month["True Consumption (MW)"], label="True", color="darkblue")
plt.plot(df_first_month["Timestamp"], df_first_month["Predicted Consumption (MW)"], label="Predicted", color="red", linestyle="--")
plt.title("Energy Consumption (First Month): Predictions vs Ground Truth")
plt.xlabel("Time")
plt.ylabel("Consumption (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "comparison_plot_1month.png")
plt.savefig(plot_path)
print(f"[Saved] 1-Month comparison plot: {plot_path}")
plt.show()
