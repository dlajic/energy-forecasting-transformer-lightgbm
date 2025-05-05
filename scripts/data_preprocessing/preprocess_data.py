import pandas as pd
import numpy as np
import os
import logging
from scripts.config_main import RAW_DATA_PATH, DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info(f"RAW_DATA_PATH: {RAW_DATA_PATH}")
logging.info(f"DATA_PATH: {DATA_PATH}")


def load_data(filepath):
    """Loads raw data from a CSV file."""
    df = pd.read_csv(filepath, sep=',')
    return df

def preprocess_data(df):
    """Performs all necessary data preprocessing steps."""

    logging.info(f"Columns in input df: {df.columns.tolist()}")

    # Ensure "datetime" is recognized as datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # interpolate nas in temp and chek after if these make sense
    df.set_index("datetime", inplace=True)

    # Interpolation
    df["temperature_c"] = df["temperature_c"].interpolate(method="time")

    # Index zurücksetzen (falls nötig)
    df.reset_index(inplace=True)

    # Duplikate in 'date' erkennen – nur einen behalten
    df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)

    # Rename columns
    df.rename(columns={'datetime': 'date'}, inplace=True)

    # we want to start with a day at 00:00:00, delete the hours before (want complete day, also because of temp)
    # find first day that starts with 00:00:00
    first_midnight_idx = df[df["date"].dt.time == pd.to_datetime("00:00:00").time()].index[0]

    # cut the lines before
    df = df.loc[first_midnight_idx:].reset_index(drop=True)

    # Feature Engineering: Create time-based features
    df['hour_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['date']).dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['date']).dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['date']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['date']).dt.month / 12)

        # Wochentag-Sin/Cos
    df['weekday'] = df['date'].dt.weekday
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df.drop(columns=['weekday'], inplace=True)

    # Gleitender Mittelwert über 6 Stunden
    df['rolling_mean_6h'] = df['consumption_MW'].rolling(window=6).mean()

    # Lag-Features
    df["consumption_last_hour"] = df["consumption_MW"].shift(1)
    df["consumption_yesterday"] = df["consumption_MW"].shift(24)
    df["consumption_last_week"] = df["consumption_MW"].shift(168)

    # Drop NaNs (wegen rolling/lags)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def save_data(df, output_path):
    """Saves the processed data to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Data saved to: {output_path}")

if __name__ == "__main__":
    logging.info("Loading raw data...")
    df_raw = load_data(RAW_DATA_PATH)

    logging.info("Preprocessing data...")
    df_processed = preprocess_data(df_raw)

    logging.info("Saving processed data...")
    save_data(df_processed, DATA_PATH)

    logging.info("Data preparation complete.")
