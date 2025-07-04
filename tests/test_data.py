import pandas as pd
from streamlit_simulation.utils_streamlit import load_data  # oder dorthin, wo die Funktion liegt

def test_load_data():
    df = load_data()
    assert not df.empty
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
