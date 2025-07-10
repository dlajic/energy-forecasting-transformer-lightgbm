import os

import pandas as pd
from huggingface_hub import hf_hub_download

from streamlit_simulation.config_streamlit import DATA_PATH

HF_REPO = "dlaj/energy-forecasting-files"
HF_FILENAME = "data/processed/energy_consumption_aggregated_cleaned.csv"


def load_data():
    # Prüfe, ob lokale Datei existiert
    if not os.path.exists(DATA_PATH):
        print(f"Lokale Datei nicht gefunden: {DATA_PATH}")
        print("Lade von Hugging Face...")

        # Lade von HF Hub
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            repo_type="dataset",
            cache_dir="hf_cache",  # Optional: lokaler Cache-Ordner
        )

        return pd.read_csv(downloaded_path, parse_dates=["date"])

    print(f"Lade lokale Datei: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def resolve_csv_path() -> str:
    if os.path.exists(DATA_PATH):
        print(f"Lokale Datei verwendet: {DATA_PATH}")
        return DATA_PATH
    else:
        print(f"Lokale Datei nicht gefunden, lade von HF: {HF_FILENAME}")
        return hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            repo_type="dataset",
            cache_dir="hf_cache",
        )
