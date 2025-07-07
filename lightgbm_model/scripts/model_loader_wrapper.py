import os
from dotenv import load_dotenv
from lightgbm_model.scripts.utils import load_lightgbm_model as real_model

# Optional: Nur wenn nicht schon global in app.py geladen
load_dotenv()

USE_DUMMY = os.getenv("USE_DUMMY_MODEL", "false").lower() == "true"

def load_lightgbm_model():
    if USE_DUMMY:
        from streamlit_simulation.dummy import DummyLightGBMModel
        return DummyLightGBMModel()
    else:
        return real_model()