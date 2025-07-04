# utils/data_utils.py
import pandas as pd
from streamlit_simulation.config_streamlit import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df
