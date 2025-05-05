# config_main.py
import os

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "energy_consumption_raw.csv")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "energy_consumption_aggregated_cleaned.csv")

# === External Temperature Data ===
TEMPERATURE_RAW_PATH = os.path.join(BASE_DIR, "data", "external", "Temperature_chicago.csv")
TEMPERATURE_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "external", "temperature_chicago_preprocessed.csv")
ENERGY_RAW_COMED_PATH = os.path.join(BASE_DIR, "data", "raw", "COMED_hourly.csv")
MERGED_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "energy_consumption_raw.csv")
