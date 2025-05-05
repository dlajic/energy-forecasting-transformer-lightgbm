# config.py
import os

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "energy_consumption_aggregated_cleaned.csv")

# Other  paths
CHECKPOINT_DIR = os.path.join(BASE_DIR, "model", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ========== Model Settings ==========
SEQ_LEN = 512               # Input sequence length (number of time steps the model sees)
FORECAST_HORIZON = 1        # Number of future steps the model should predict
HEAD_DROPOUT = 0.1          # Dropout in the head to prevent overfitting
WEIGHT_DECAY = 0.0          # L2 regularization (0 means off)

# ========== Training Settings ==========
MAX_EPOCHS = 9              # Optimal number of epochs based on performance curve
BATCH_SIZE = 32             # Batch size for training and evaluation
LEARNING_RATE = 1e-4        # Base learning rate
MAX_LR = 1e-4               # Max LR for OneCycleLR scheduler
GRAD_CLIP = 5.0             # Gradient clipping threshold

# ========== Freezing Strategy ==========
FREEZE_ENCODER = True
FREEZE_EMBEDDER = True
FREEZE_HEAD = False         #just unfreeze the last forecasting head for finetuning
