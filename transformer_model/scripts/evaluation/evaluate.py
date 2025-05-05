# evaluate.py

import os
import json
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, r2_score

from transformer_model.scripts.config_transformer import BASE_DIR, RESULTS_DIR, CHECKPOINT_DIR, DATA_PATH, FORECAST_HORIZON, SEQ_LEN
from transformer_model.scripts.training.load_basis_model import load_moment_model
from transformer_model.scripts.utils.informer_dataset_class import InformerDataset
from momentfm.utils.utils import control_randomness
from transformer_model.scripts.utils.check_device import check_device


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate():
    control_randomness(seed=13)
    # Set device
    device, backend, scaler = check_device()
    logging.info(f"Evaluation is running on: {backend} ({device})")

    # Load final model
    model = load_moment_model()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)  
    model.eval()
    logging.info(f"Loaded final model from: {checkpoint_path}")

    # Recreate training dataset to get the fitted scaler
    train_dataset = InformerDataset(
        data_split="train",
        random_seed=13,
        forecast_horizon=FORECAST_HORIZON
    )

    # Use its scaler in the test dataset
    test_dataset = InformerDataset(
        data_split="test",
        random_seed=13,
        forecast_horizon=FORECAST_HORIZON
    )

    test_dataset.scaler = train_dataset.scaler

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    trues, preds = [], []

    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(test_loader, desc="Evaluating on test set"):
            timeseries = timeseries.float().to(device)
            forecast = forecast.float().to(device)
            input_mask = input_mask.to(device)  # <- wichtig!

            output = model(x_enc=timeseries, input_mask=input_mask)

            trues.append(forecast.cpu().numpy())
            preds.append(output.forecast.cpu().numpy())


    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)

    # Extract only first feature (consumption)
    true_values = trues[:, 0, :]
    pred_values = preds[:, 0, :]

    # Inverse normalization
    n_features = test_dataset.n_channels
    true_reshaped = np.column_stack([true_values.flatten()] + [np.zeros_like(true_values.flatten())] * (n_features - 1))
    pred_reshaped = np.column_stack([pred_values.flatten()] + [np.zeros_like(pred_values.flatten())] * (n_features - 1))

    true_original = test_dataset.scaler.inverse_transform(true_reshaped)[:, 0]
    pred_original = test_dataset.scaler.inverse_transform(pred_reshaped)[:, 0]


    # Build timestamp index, since date got cutted out in informerdataset we need original dataset and use the index of the beginning of testdata to get the date
    csv_path = os.path.join(DATA_PATH) 
    df = pd.read_csv(csv_path, parse_dates=["date"])

    train_len = len(train_dataset)
    test_start_idx = train_len + SEQ_LEN
    start_timestamp = df["date"].iloc[test_start_idx]
    logging.info(f"[DEBUG] timestamp: {start_timestamp}")

    timestamps = [start_timestamp + pd.Timedelta(hours=i) for i in range(len(true_original))]

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "True Consumption (MW)": true_original,
        "Predicted Consumption (MW)": pred_original
    })

    # Save results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "test_results.csv")
    df.to_csv(results_path, index=False)
    logging.info(f"Saved prediction results to: {results_path}")

    # Evaluation metrics
    mse = mean_squared_error(df["True Consumption (MW)"], df["Predicted Consumption (MW)"])
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((df["True Consumption (MW)"] - df["Predicted Consumption (MW)"]) / df["True Consumption (MW)"])) * 100
    r2 = r2_score(df["True Consumption (MW)"], df["Predicted Consumption (MW)"])

    # Save metrics to JSON
    metrics = {"RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    logging.info(f"Saved evaluation metrics to: {metrics_path}")
    logging.info(f"RMSE: {rmse:.3f} | MAPE: {mape:.2f}% | R²: {r2:.3f}")

if __name__ == "__main__":
    evaluate()
