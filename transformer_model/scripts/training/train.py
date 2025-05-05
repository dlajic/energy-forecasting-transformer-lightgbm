# train.py

import os
import json
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from transformer_model.scripts.config_transformer import (
    BASE_DIR,
    MAX_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_LR,
    GRAD_CLIP,
    FORECAST_HORIZON,
    CHECKPOINT_DIR,
    RESULTS_DIR
)

from transformer_model.scripts.training.load_basis_model import load_moment_model
from transformer_model.scripts.utils.create_dataloaders import create_dataloaders
from transformer_model.scripts.utils.check_device import check_device
from momentfm.utils.utils import control_randomness


# === Setup logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train():
    # Start timing
    start_time = time.time()

    # Setup device (CUDA / DirectML / CPU) and AMP scaler
    device, backend, scaler = check_device()

    # Load base model
    model = load_moment_model().to(device)

    # Set random seeds for reproducibility
    control_randomness(seed=13)

    # Setup loss function and optimizer
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load data
    train_loader, test_loader = create_dataloaders()

    # Setup learning rate scheduler (OneCycle policy)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=total_steps,
        pct_start=0.3
    )

    # Ensure output folders exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Store metrics
    train_losses, test_mses, test_maes = [], [], []

    best_mae = float("inf")
    best_epoch = None
    no_improve_epochs = 0
    patience = 5  

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_losses = []

        for timeseries, forecast, input_mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass (with AMP if enabled)
            if scaler:
                with torch.amp.autocast(device_type="cuda"):
                    output = model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)
            else:
                output = model(x_enc=timeseries, input_mask=input_mask)
                loss = criterion(output.forecast, forecast)

            # Backward pass + optimization
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            epoch_losses.append(loss.item())

        average_train_loss = np.mean(epoch_losses)
        train_losses.append(average_train_loss)
        logging.info(f"Epoch {epoch}: Train Loss = {average_train_loss:.4f}")

        # === Evaluation ===
        model.eval()
        trues, preds = [], []

        with torch.no_grad():
            for timeseries, forecast, input_mask in test_loader:
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                if scaler:
                    with torch.amp.autocast(device_type="cuda"):
                        output = model(x_enc=timeseries, input_mask=input_mask)
                else:
                    output = model(x_enc=timeseries, input_mask=input_mask)

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)


        # Reshape for sklearn metrics
        trues_2d = trues.reshape(trues.shape[0], -1)
        preds_2d = preds.reshape(preds.shape[0], -1)

        mse = mean_squared_error(trues_2d, preds_2d)
        mae = mean_absolute_error(trues_2d, preds_2d)

        test_mses.append(mse)
        test_maes.append(mae)
        logging.info(f"Epoch {epoch}: Test MSE = {mse:.4f}, MAE = {mae:.4f}")

        # === Early Stopping Check ===
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            no_improve_epochs = 0

            # Save best model
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to: {best_model_path} (MAE: {best_mae:.4f})")
        else:
            no_improve_epochs += 1
            logging.info(f"No improvement in MAE for {no_improve_epochs} epoch(s).")

            if no_improve_epochs >= patience:
                logging.info("Early stopping triggered.")
                break

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        scheduler.step()

    logging.info(f"Best model was at epoch {best_epoch} with MAE: {best_mae:.4f}")

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
    logging.info(f"Final Test MSE: {test_mses[-1]:.4f}, MAE: {test_maes[-1]:.4f}")

    # Save training metrics
    metrics = {
        "train_losses": [float(x) for x in train_losses],
        "test_mses": [float(x) for x in test_mses],
        "test_maes": [float(x) for x in test_maes]
    }

    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    logging.info(f"Training metrics saved to: {metrics_path}")

    # Done
    elapsed = time.time() - start_time
    logging.info(f"Training complete in {elapsed / 60:.2f} minutes.")


# === Entry Point ===
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
