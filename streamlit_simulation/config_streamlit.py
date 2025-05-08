# config_streamlit
import os

# Base directory → points to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model paths
MODEL_PATH_LIGHTGBM = os.path.join(BASE_DIR, "lightgbm_model", "model", "lightgbm_final_model.pkl")
MODEL_PATH_TRANSFORMER = os.path.join(BASE_DIR, "transformer_model", "model", "checkpoints", "model_final.pth")

# Data path
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "energy_consumption_aggregated_cleaned.csv")

# Color palette for Streamlit layout
PLOT_COLOR = "#e4eaf0"          # Plot background color

# Constants
TRAIN_RATIO = 0.7  # Train/test split ratio used by both models
