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
TEXT_COLOR = "#004080"         # Primary text color (clean dark blue)
HEADER_COLOR = "#002855"       # Accent color for headings
ACCENT_COLOR = "#9bb2cc"       # For borders, highlights, etc.
BUTTON_BG = "#dee7f0"          # Background color for buttons
BUTTON_HOVER_BG = "#cbd9e6"    # Hover color for buttons
BG_COLOR = "#ffffff"           # Page background
INPUT_BG = "#f2f6fa"           # Background for select boxes, inputs
PROGRESS_COLOR = "#0077B6"     # Progress bar color
PLOT_COLOR = "white"           # Plot background color

# Constants
TRAIN_RATIO = 0.7  # Train/test split ratio used by both models
