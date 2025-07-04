import pandas as pd
import numpy as np

from lightgbm_model.scripts.utils import load_lightgbm_model
from streamlit_simulation.config_streamlit import DATA_PATH
from lightgbm_model.scripts.config_lightgbm import FEATURES

def test_lightgbm_model_load():
    model = load_lightgbm_model()
    assert model is not None

#def test_lightgbm_prediction():
#    model = load_lightgbm_model()
#    sample_input = np.random.rand(1, model.n_features_in_)  # Dummy Input
#    prediction = model.predict(sample_input)
#    assert prediction is not None
#    assert len(prediction) == 1


def test_lightgbm_real_prediction():
    # Daten laden
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    
    # Kleine Testmenge extrahieren
    sample = df[FEATURES].dropna().iloc[:1]  # 1 Beispiel, alle Features

    # Modell laden
    model = load_lightgbm_model()

    # Vorhersage
    prediction = model.predict(sample)

    # Tests
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)